# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:37:02 2020

@author: raphael.dutrieux
"""
import warnings
import plotly.graph_objects as go
import geopandas as gp
import xarray as xr
import rasterio
import rasterio.features
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib.colors as colors


# from fordead.masking_vi import get_dict_vi
from goodforest_lib.utils.compute_additional_bands import get_dict_vi

from fordead.import_data import import_resampled_sen_stack, import_soil_data, import_dieback_data, import_binary_raster, import_stress_data, import_stress_index


def get_stack_rgb(tile, extent, bands = ["B4","B3","B2"], dates = None):
    """
    Imports stack of bands, clipped with extent

    Parameters
    ----------
    tile : TileInfo object
        TileInfo objects with paths attribute which is a dictionnary containing file paths to SENTINEL (tile.paths["Sentinel"][YYYY-MM-DD] returns a dictionnary where keys are band names and values are the paths to the files)
    extent : list or 1D array, optional
        Extent used for cropping [xmin,ymin, xmax,ymax]. If None, there is no cropping. The default is None.
    bands : list, optional
        List of bands. The default is ["B4","B3","B2"].
    dates : 1D array or list
        List of dates to import, if None, all dates saved in "tile" TileInfo object are used
    Returns
    -------
    stack_rgb : xarray DataArray
        DataArray with dimensions (x,y,Time,band)

    """
    
    if dates is None:
        dates = tile.dates
    list_rgb = [import_resampled_sen_stack(tile.paths["Sentinel"][date], bands,extent = extent) for date in dates]
    stack_rgb = xr.concat(list_rgb,dim="Time")
    stack_rgb=stack_rgb.assign_coords(Time=dates)
    stack_rgb=stack_rgb.transpose("Time", 'y', 'x',"band")
    return stack_rgb

def polygon_from_coordinate_and_radius(coordinates, radius, crs):
    """
    Creates rectangular polygon from coordinates and a radius

    Parameters
    ----------
    coordinates : list or tuple
        [x,y] list
    radius : int
    crs : str
        crs as understood by geopandas.GeoDataFrame

    Returns
    -------
    polygon : geopandas.GeoDataFrame

    """
    
    lon_point_list = [coordinates[0]-radius,coordinates[0]+radius,coordinates[0]+radius,coordinates[0]-radius,coordinates[0]-radius,coordinates[0]-radius]
    lat_point_list = [coordinates[1]+radius,coordinates[1]+radius,coordinates[1]-radius,coordinates[1]-radius,coordinates[1]+radius,coordinates[1]-radius]
    
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    polygon = gp.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
    return polygon

def intersects_area(shape, extent, crs):
    polygon_geom = Polygon(zip([extent[0],extent[2],extent[2],extent[0],extent[0]], 
                               [extent[1],extent[1],extent[3],extent[3],extent[1]]))
    polygon = gp.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom]).reset_index(drop=True)   
    intersects_bool = polygon.intersects(shape.reset_index(drop=True))[0]
    return intersects_bool
    

def CreateTimelapse(shape,tile,vector_display_path, hover_column_list, max_date, show_confidence_class):
        nb_classes = len(tile.parameters["conf_classes_list"]) if show_confidence_class else 1
        NbData = 1 + nb_classes + 2*tile.parameters["soil_detection"]
        color_map = plt.get_cmap('Reds',nb_classes+1)
        
        fig = go.Figure()
        
        
        #Récupération des données raster
        extent = shape.total_bounds
        
        if tile.parameters["soil_detection"]:
            soil_data = import_soil_data(tile.paths)
            soil_data = soil_data.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
        if show_confidence_class: 
            stress_data = import_stress_data(tile.paths)
            stress_index = import_stress_index(tile.paths["stress_index"])
            confidence_index = stress_index.sel(period = (stress_data["nb_periods"]+1).where(stress_data["nb_periods"]<=tile.parameters["max_nb_stress_periods"],tile.parameters["max_nb_stress_periods"]))
            nb_dates = stress_data["nb_dates"].sel(period = (stress_data["nb_periods"]+1).where(stress_data["nb_periods"]<=tile.parameters["max_nb_stress_periods"],tile.parameters["max_nb_stress_periods"]))
            
            confidence_index = confidence_index.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
            nb_dates = nb_dates.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
            digitized_confidence = np.digitize(confidence_index,np.array(tile.parameters["conf_threshold_list"]).astype(float))
            digitized_confidence[nb_dates==3]=0
            
        dieback_data = import_dieback_data(tile.paths)
        dieback_data = dieback_data.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
      
        forest_mask = import_binary_raster(tile.paths["forest_mask"]).loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
        sufficient_coverage_mask = import_binary_raster(tile.paths["sufficient_coverage_mask"]).loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
        if tile.parameters["stress_index_mode"] is not None:
            too_many_stress_periods_mask = import_binary_raster(tile.paths["too_many_stress_periods_mask"]).loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
            relevant_area = forest_mask & sufficient_coverage_mask & too_many_stress_periods_mask
        else:
            relevant_area = forest_mask & sufficient_coverage_mask
            
        #Correcting extent if computed area is smaller than Sentinel-2 data area
        extent = np.array([float(forest_mask[dict(x=0,y=0)].coords["x"])-forest_mask.rio.transform()[0]/2,
                                                float(forest_mask[dict(x=-1,y=-1)].coords["y"])-forest_mask.rio.transform()[0]/2,
                                                float(forest_mask[dict(x=-1,y=-1)].coords["x"])+forest_mask.rio.transform()[0]/2,
                                                float(forest_mask[dict(x=0,y=0)].coords["y"])+forest_mask.rio.transform()[0]/2])
        
        dates = tile.dates[tile.dates <= max_date] if max_date is not None else tile.dates
        stack_rgb = get_stack_rgb(tile, extent, bands = ["B4","B3","B2"], dates = dates)


        # CountInvalid=0
        valid_dates = xr.where(stack_rgb == -10000,True,False).sum(dim="x").sum(dim="y").sum(dim="band")/(stack_rgb.size / stack_rgb.sizes["Time"]) < 0.90
        nb_valid_dates = int(valid_dates.sum())
        for dateIndex, date in enumerate(dates):
            if valid_dates.sel(Time = date) == True:
                fig.add_trace(
                    go.Image(
                        visible=False,
                        z=stack_rgb.sel(Time = date).data,
                        name=date,
                        zmin=[0, 0, 0, 0], zmax=[600,600,600,600]))
                
                # Labels+=[date]
                DictCoordX={}
                DictCoordY={}
                for etat in range(1,nb_classes + 2*tile.parameters["soil_detection"]+1):
                    DictCoordX[etat] = []
                    DictCoordY[etat] = []

                detected = (dieback_data["first_date"] <= dateIndex) & dieback_data["state"]
                if tile.parameters["soil_detection"]:
                    soil = (soil_data["first_date"] <= dateIndex) & soil_data["state"]
                    if show_confidence_class:
                        # affected = detected.where(~soil, len(tile.parameters["conf_classes_list"])+detected+soil)
                        affected = xr.where(~soil, detected*(1+digitized_confidence),len(tile.parameters["conf_classes_list"])+detected+soil)
                    else:
                        affected=detected+2*soil
                elif show_confidence_class:
                    affected = detected*(1+digitized_confidence)
                else:
                    affected=detected
                
                # valid_area = 
                affected = affected.where(relevant_area,0)
                
                results_affected = (
                            {'properties': {'Etat': v}, 'geometry': s}
                            for i, (s, v) 
                            in enumerate(
                                rasterio.features.shapes(affected.data.astype("uint8"), transform=stack_rgb.rio.transform())))
                
                geomsaffected = list(results_affected)
                for geom in geomsaffected:
                    if geom["properties"]["Etat"] != 0:
#                            print(geom["geometry"]["coordinates"])
                        for poly in geom["geometry"]["coordinates"]:
                            xList=np.array([coord[0] for coord in poly])
                            yList=np.array([coord[1] for coord in poly])
                            
                            DictCoordX[int(geom["properties"]["Etat"])]+=list((xList-np.array(stack_rgb.rio.transform()[2]))/10-0.5)+[None]
                            DictCoordY[int(geom["properties"]["Etat"])]+=list((np.array(stack_rgb.rio.transform()[5])-yList)/10-0.5)+[None]
                


                for etat in range(1,nb_classes+1):
                    fig.add_trace(go.Scatter(
                        x=DictCoordX[etat],
                        y=DictCoordY[etat],
                        line_color=colors.rgb2hex(color_map(etat-1)),
                        hoverinfo="skip",
                        name='Dieback detected' if not(show_confidence_class) else tile.parameters["conf_classes_list"][etat-1],
                        legendgroup="dieback",
                        legendgrouptitle_text="Dieback detected",
                        
                        ))
                    
                if tile.parameters["soil_detection"]:
                    fig.add_trace(go.Scatter(
                        x=DictCoordX[etat+1],
                        y=DictCoordY[etat+1],
                        line_color="black",
                        hoverinfo="skip",
                        name='Bare ground',
                        legendgroup="bare_ground",
                        legendgrouptitle_text="Bare ground detected",
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=DictCoordX[etat+2],
                        y=DictCoordY[etat+2],
                        line_color="blue",
                        line_width=3,
                        hoverinfo="skip",
                        name='Bare ground after dieback',
                        legendgroup="bare_ground"
                    ))
                
            # else:
            #     CountInvalid+=1
        
        
        for data in range(len(fig.data)):
            fig.data[data].visible = False
        fig.data[0].visible = True
        
        
        # Visualisation vector display
        if vector_display_path is not None:
            
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sequential read of iterator was interrupted. Resetting iterator. This can negatively impact the performance.")
                vector_display = gp.read_file(vector_display_path,bbox=shape.envelope).to_crs(crs=stack_rgb.rio.crs).explode(index_parts=True)
            nb_vector_obj = vector_display.shape[0]
            
            if type(hover_column_list) is str: hover_column_list = [hover_column_list]
            
            for column in hover_column_list:
                if column not in vector_display.columns:
                    raise Exception(vector_display_path + " has no column '" + column + "'")
            
            for obj_index in range(nb_vector_obj):
                obj = vector_display.iloc[obj_index]

                if obj['geometry'].geom_type == "Point":
                    mode = "markers"
                    coords=obj['geometry'].coords.xy
                elif obj['geometry'].geom_type == "LineString":
                    mode = "lines"
                    coords=obj['geometry'].coords.xy
                elif obj['geometry'].geom_type == "Polygon":
                    mode = "lines"
                    coords=obj['geometry'].exterior.coords.xy
                else:
                    raise Exception("geom_type " + obj['geometry'].geom_type + " not supported")
                    
                x1=(coords[0]-np.array(int(stack_rgb.x.min())-5))/10-0.5
                y1=(np.array(int(stack_rgb.y.max())+5)-coords[1])/10-0.5

                fig.add_trace(go.Scatter(
                x=x1,
                y=y1, 
                mode=mode,
                marker=dict(
                    color='white',
                    size=10,
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                # marker_color = "white",
                # marker_size = 12,
                # marker_line_width = 5,
                # marker_line_color="darkviolet",
                line_color="darkviolet",
                name=" | ".join([str(obj[column]) for column in hover_column_list]),
                hovertemplate=" | ".join([str(obj[column]) for column in hover_column_list]),
                legendgroup="vector_display",
                legendgrouptitle_text=" | ".join(hover_column_list)
                ))
        else:
            nb_vector_obj = 0
            
        results_affected = (
                    {'properties': {'Etat': v}, 'geometry': s}
                    for i, (s, v) 
                    in enumerate(
                        rasterio.features.shapes(relevant_area.data.astype("uint8"), transform=stack_rgb.rio.transform())))
        dict_relevant_area = {"x" : [],
                            "y" : []}
        for geom in list(results_affected):
            if geom["properties"]["Etat"] == 0:
                for poly in geom["geometry"]["coordinates"]:
                    xList=np.array([coord[0] for coord in poly])
                    yList=np.array([coord[1] for coord in poly])
                    
                    dict_relevant_area["x"] +=list((xList-np.array(stack_rgb.rio.transform()[2]))/10-0.5)+[None]
                    dict_relevant_area["y"] +=list((np.array(stack_rgb.rio.transform()[5])-yList)/10-0.5)+[None]
                    
        fig.add_trace(go.Scatter(
            x=dict_relevant_area["x"],
            y=dict_relevant_area["y"],
            fill = "none",
            hoveron = 'fills',
            line_color = "grey",
            name="Permanently masked pixels"))
        
        
        #Slider  
        steps = []
        for i in range(nb_valid_dates):
            
            step = dict(
                label = dates[valid_dates][i],
                method="restyle",
                args=["visible", [False]*nb_valid_dates*NbData + [True]*nb_vector_obj + [True]])
                # args=["visible", [False] * (len(Day)+stackAtteint.shape[0]*3) + [True] * scolytes.shape[0] + [False] * stackMask.shape[0]] ,
            
            for n in range(NbData):
                step["args"][1][i*NbData+n] = True  # Toggle i*NbData+n'th trace to "visible"
            steps.append(step)
            
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Date : "},
            pad={"t": 50},
            steps=steps
        )]
        
        try:
            fig.update_layout(
                sliders=sliders,
                margin_autoexpand=False,
                margin_b = 130,
                margin_r = 250,
                legend_groupclick = "toggleitem")
        except:
            print("If this message appears, your plotly package version might be under 5.3 and you are advised to update it")
            fig.update_layout(
                sliders=sliders,
                margin_autoexpand=False,
                margin_b = 130,
                margin_r = 250)
        
        fig.update_xaxes(range=[0, stack_rgb.shape[1]])
        fig.update_yaxes(range=[stack_rgb.shape[2],0])
        fig.update_layout(showlegend=True)
        
    
        return fig


def plot_mask(pixel_series, xy_soil_data, xy_dieback_data, xy_first_detection_date_index, X,Y, yy, threshold_anomaly, vi, path_dict_vi,ymin,ymax):
    """
    Creates figure from all data

    """
    pixel_series.mask = pixel_series.mask.compute()
    fig=plt.figure(figsize=(10,7))
    ax = plt.gca()
    formatter = mdates.DateFormatter("%b %Y")
    ax.xaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=30)
    if pixel_series.Soil.any() : pixel_series.where(pixel_series.Soil,drop=True).plot.line('k^', label='Coupe') 
    
    if xy_first_detection_date_index !=0:
        pixel_series.where(pixel_series.training_date & ~pixel_series.mask,drop=True).plot.line("bo", label='Training')
        if (~pixel_series.training_date & ~pixel_series.Anomaly).any() : pixel_series.where(~pixel_series.training_date & ~pixel_series.Anomaly,drop=True).plot.line("o",color = '#1fca3b', label='Detection dates') 
  
        
    else:
        if (~pixel_series.mask).any(): pixel_series.where(~pixel_series.mask,drop=True).plot.line("bo")
        plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\n Not enough dates to compute a model",size=15)
         
    [plt.axvline(x=datetime.datetime.strptime(str(year)+"-01-01", '%Y-%m-%d'),color="black") for year in np.arange(pixel_series.isel(Time = 0).Time.data.astype('datetime64[Y]'), (pixel_series.isel(Time = -1).Time.data + np.timedelta64(35,'D')).astype('datetime64[Y]')+1) if str(year)!="2015"]

    plt.legend()
    plt.ylim((ymin,ymax))
    plt.xlabel("Date",size=15)
    plt.ylabel(vi,size=15)
        
    return fig

def plot_model(pixel_series, xy_soil_data, xy_dieback_data, xy_first_detection_date_index, X,Y, yy, threshold_anomaly, vi, path_dict_vi,ymin,ymax):
    """
    Creates figure from all data

    """
    pixel_series["mask"] = pixel_series.mask.compute()
    fig=plt.figure(figsize=(10,7))
    ax = plt.gca()
    formatter = mdates.DateFormatter("%b %Y")
    ax.xaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=30)
    if pixel_series.Soil.any() : pixel_series.where(pixel_series.Soil,drop=True).plot.line('k^', label='Coupe') 
    
    if xy_first_detection_date_index !=0:
        pixel_series.where(pixel_series.training_date & ~pixel_series.mask,drop=True).plot.line("bo", label='Training dates')
        if (~pixel_series.training_date & ~pixel_series.mask).any() : pixel_series.where(~pixel_series.training_date & ~pixel_series.mask,drop=True).plot.line("o",color = '#1fca3b', label='Detection dates') 
        # if (~pixel_series.training_date & pixel_series.Anomaly & ~pixel_series.mask).any() : pixel_series.where(~pixel_series.training_date & pixel_series.Anomaly & ~pixel_series.mask & ~pixel_series.Soil,drop=True).plot.line("r*", markersize=9, label='Dates avec anomalies') 
  
        #Plotting vegetation index model and anomaly threshold
        yy.plot.line("b", label='Vegetation index model based on training dates')
        plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y)),size=15)
        
    else:
        if (~pixel_series.mask).any(): pixel_series.where(~pixel_series.mask,drop=True).plot.line("bo")
        plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\n Not enough dates to compute a model",size=15)
         
    [plt.axvline(x=datetime.datetime.strptime(str(year)+"-01-01", '%Y-%m-%d'),color="black") for year in np.arange(pixel_series.isel(Time = 0).Time.data.astype('datetime64[Y]'), (pixel_series.isel(Time = -1).Time.data + np.timedelta64(35,'D')).astype('datetime64[Y]')+1) if str(year)!="2015"]

    plt.legend()
    plt.ylim((ymin,ymax))
    plt.xlabel("Date",size=15)
    plt.ylabel(vi,size=15)
        
    return fig

def plot_temporal_series(pixel_series, xy_soil_data, xy_dieback_data, xy_first_detection_date_index, xy_stress_data, X,Y, yy, threshold_anomaly, vi, path_dict_vi,ymin,ymax, ignored_period = None):
    """
    Creates figure from all data

    """
    pixel_series["mask"] = pixel_series.mask.compute()
    fig=plt.figure(figsize=(10,7))
    ax = plt.gca()
    formatter = mdates.DateFormatter("%b %Y")
    ax.xaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=30)
    if pixel_series.Soil.any() : pixel_series.where(pixel_series.Soil,drop=True).plot.line('k^', label='Coupe') 
    
    if xy_first_detection_date_index !=0:
        pixel_series.where(pixel_series.training_date & ~pixel_series.mask.compute(),drop=True).plot.line("bo", label='Training dates')
        if (~pixel_series.training_date & ~pixel_series.Anomaly & ~pixel_series.mask).any() : pixel_series.where(~pixel_series.training_date & ~pixel_series.Anomaly & ~pixel_series.mask,drop=True).plot.line("o",color = '#1fca3b', label='Normal dates') 
        if (~pixel_series.training_date & pixel_series.Anomaly & ~pixel_series.mask).any() : pixel_series.where(~pixel_series.training_date & pixel_series.Anomaly & ~pixel_series.mask & ~pixel_series.Soil,drop=True).plot.line("r*", markersize=9, label='Anomalies') 
  
        #Plotting vegetation index model and anomaly threshold
        
        if ignored_period is not None:
            kept_dates = ~np.array([(date.astype(str)[5:] > min(ignored_period) and date.astype(str)[5:] < max(ignored_period)) for date in yy.Time.data])
            yy = xr.DataArray(np.ma.masked_where(kept_dates, yy), coords={"Time" : yy.Time},dims=["Time"])
            
        yy.plot.line("b", label='Vegetation index model based on training dates')
        
        dict_vi = get_dict_vi(path_dict_vi)
        if dict_vi[vi]["dieback_change_direction"] == "+":
            (yy+threshold_anomaly).plot.line("b--", label='Threshold for anomaly detection')
        elif dict_vi[vi]["dieback_change_direction"] == "-":
            (yy-threshold_anomaly).plot.line("b--", label='Threshold for anomaly detection')
        
        
        if xy_stress_data is not None:
            for period in range(min(xy_stress_data.sizes["period"], int(xy_stress_data["nb_periods"]))):
                period_dates = (pixel_series.Time[xy_stress_data["date"].isel(change = [period*2,period*2+1])]).data
                label = {"label" : "Stress period"} if period==0 else {}
                plt.axvspan(xmin = period_dates[0],xmax = period_dates[1],color = "orange", alpha = 0.3, **label)         
                # plt.axvspan(xmin = period_dates[0],xmax = period_dates[1],color = "orange", alpha = 0.3, label = "Stress period")            
        
        # Plotting vertical lines when dieback or soil is detected
        if ~xy_dieback_data["state"] & ~xy_soil_data["state"]:
            plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\nHealthy pixel",size=15)
        elif xy_dieback_data["state"] & ~xy_soil_data["state"]:
            date_atteint = pixel_series.Time[int(xy_dieback_data["first_date"])].data
            plt.axvline(x=date_atteint, color='red', linewidth=3, linestyle=":",label="Dieback detection")
            plt.title("X : " + str(int(pixel_series.x))+"    Y : " + str(int(pixel_series.y))+"\nDieback detected, first anomaly on " + str(date_atteint.astype("datetime64[D]")),size=15)
        elif xy_dieback_data["state"] & xy_soil_data["state"]:
            date_atteint = pixel_series.Time[int(xy_dieback_data["first_date"])].data
            date_coupe = pixel_series.Time[int(xy_soil_data["first_date"])].data
            plt.axvline(x=date_atteint, color="red", linewidth=3, linestyle=":")
            plt.axvline(x=date_coupe,color='black', linewidth=3, linestyle=":")
            plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\nDieback detected, first anomaly on " + str(date_atteint.astype("datetime64[D]")) + ", clear cut on " + str(date_coupe.astype("datetime64[D]")),size=15)
        else:
            date_coupe = pixel_series.Time[int(xy_soil_data["first_date"])].data
            plt.axvline(x=date_coupe, color='black', linewidth=3, linestyle=":",label="Détection de coupe")
            plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\nBare ground detected on " + str(date_coupe.astype("datetime64[D]")),size=15)
    else:
        if (~pixel_series.mask).any(): pixel_series.where(~pixel_series.mask,drop=True).plot.line("bo")
        plt.title("X : " + str(int(pixel_series.x))+"   Y : " + str(int(pixel_series.y))+"\n Not enough dates to compute a model",size=15)
         
    [plt.axvline(x=datetime.datetime.strptime(str(year)+"-01-01", '%Y-%m-%d'),color="black") for year in np.arange(pixel_series.isel(Time = 0).Time.data.astype('datetime64[Y]'), (pixel_series.isel(Time = -1).Time.data + np.timedelta64(35,'D')).astype('datetime64[Y]')+1) if str(year)!="2015"]

    plt.legend()
    plt.ylim((ymin,ymax))
    plt.xlabel("Date",size=15)
    plt.ylabel(vi,size=15)
        
    return fig

def select_pixel_from_coordinates(X,Y, harmonic_terms, coeff_model, first_detection_date_index, soil_data, dieback_data, stack_masks, stack_vi, anomalies):
    
    xy_first_detection_date_index = int(first_detection_date_index.sel(x = X, y = Y,method = "nearest"))
    xy_soil_data = soil_data.sel(x = X, y = Y,method = "nearest")
    xy_stack_masks = stack_masks.sel(x = X, y = Y,method = "nearest")
    pixel_series = stack_vi.sel(x = X, y = Y,method = "nearest")
    if xy_first_detection_date_index!=0:
        xy_anomalies = anomalies.sel(x = X, y = Y,method = "nearest").compute()
        xy_dieback_data = dieback_data.sel(x = X, y = Y,method = "nearest")
        anomalies_time = xy_anomalies.Time.where(xy_anomalies,drop=True).astype("datetime64[D]").data
        pixel_series = pixel_series.assign_coords(Anomaly = ("Time", [time in anomalies_time for time in stack_vi.Time.data]))
        pixel_series = pixel_series.assign_coords(training_date=("Time", [index < xy_first_detection_date_index for index in range(pixel_series.sizes["Time"])]))
        yy = (harmonic_terms * coeff_model.sel(x = X, y = Y,method = "nearest").compute()).sum(dim="coeff")
    else:
        xy_dieback_data=None
        xy_anomalies = None
        yy = None
    pixel_series = pixel_series.assign_coords(Soil = ("Time", [index >= int(xy_soil_data["first_date"]) if xy_soil_data["state"] else False for index in range(pixel_series.sizes["Time"])]))
    pixel_series = pixel_series.assign_coords(mask = ("Time", xy_stack_masks.data))
        
    return pixel_series, yy,  xy_soil_data, xy_dieback_data, xy_first_detection_date_index
    

def select_pixel_from_indices(X,Y, harmonic_terms, coeff_model, first_detection_date_index, soil_data, dieback_data, stack_masks, stack_vi, anomalies,stress_data):
    yy = (harmonic_terms * coeff_model.isel(x = X, y = Y).compute()).sum(dim="coeff")
    
    xy_first_detection_date_index = int(first_detection_date_index.isel(x = X, y = Y))
    xy_soil_data = soil_data.isel(x = X, y = Y) if soil_data is not None else {"state" : False}
    xy_stack_masks = stack_masks.isel(x = X, y = Y)
    pixel_series = stack_vi.isel(x = X, y = Y)
    if stress_data is not None:
        xy_stress_data = stress_data.isel(x = X, y = Y)
    else:
        xy_stress_data = None
        
    if xy_first_detection_date_index!=0:
        xy_anomalies = anomalies.isel(x = X, y = Y).compute()
        xy_dieback_data = dieback_data.isel(x = X, y = Y)
    else:
        xy_dieback_data=None
        xy_anomalies = None
    pixel_series = pixel_series.assign_coords(Soil = ("Time", [index >= int(xy_soil_data["first_date"]) if xy_soil_data["state"] else False for index in range(pixel_series.sizes["Time"])]))
    pixel_series = pixel_series.assign_coords(mask = ("Time", xy_stack_masks.data))
    if xy_first_detection_date_index!=0:
        anomalies_time = xy_anomalies.Time.where(xy_anomalies,drop=True).astype("datetime64[D]").data
        pixel_series = pixel_series.assign_coords(Anomaly = ("Time", [time in anomalies_time for time in stack_vi.Time.data]))
        pixel_series = pixel_series.assign_coords(training_date=("Time", [index < xy_first_detection_date_index for index in range(pixel_series.sizes["Time"])]))
        
    return pixel_series, yy,  xy_soil_data, xy_dieback_data, xy_first_detection_date_index, xy_stress_data

def select_and_plot_time_series(x,y, forest_mask, harmonic_terms, coeff_model, first_detection_date_index, soil_data, dieback_data, stack_masks, stack_vi, anomalies, stress_data, tile, ymin, ymax, name_file = None):

    
    if x < 0 or x >= tile.raster_meta["sizes"]["x"] or y < 0 or y >= tile.raster_meta["sizes"]["y"]:
        print("Pixel outside extent of the region of interest")
    else:
        x = int(x)
        y = int(y)
        xy_forest_mask = forest_mask.isel(x = x, y = y)
        if not(xy_forest_mask):
            print("Pixel outside forest mask")
        else:
            pixel_series, yy,  xy_soil_data, xy_dieback_data, xy_first_detection_date_index, xy_stress_data = select_pixel_from_indices(x,y, harmonic_terms, coeff_model, first_detection_date_index, soil_data, dieback_data, stack_masks, stack_vi, anomalies, stress_data)              
            if xy_stress_data is not None and xy_stress_data["nb_periods"]>tile.parameters["max_nb_stress_periods"]:
                print("Maximum number of stress periods exceeded")
            else:
                fig = plot_temporal_series(pixel_series, xy_soil_data, xy_dieback_data, xy_first_detection_date_index, xy_stress_data, x, y, yy, tile.parameters["threshold_anomaly"],tile.parameters["vi"],tile.parameters["path_dict_vi"],ymin,ymax, ignored_period = tile.parameters["ignored_period"])
    
                
                if name_file is None: name_file = "X"+str(int(pixel_series.x))+"_Y"+str(int(pixel_series.y))
                fig.savefig(tile.paths["series"] / (name_file + ".png"))
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")
                    plt.show()
                plt.close()



# def DetermineTuiles(shapeInteret):
#     Tuiles=gp.read_file(os.getcwd()+"/Data/Vecteurs/TilesSentinel.shp")
#     shapeInteret=shapeInteret.to_crs(Tuiles.crs)
#     TuilesOverlay=gp.overlay(shapeInteret,Tuiles)
#     # TuilesOverlay[TuilesOverlay["Name"]=="31UFQ"]
#     CountOverlay=TuilesOverlay["Name"].value_counts().sort_values(ascending=False)
#     TuilesOverlay['Name']=pd.Categorical(TuilesOverlay['Name'], list(CountOverlay.index)) #Donne l'ordre des tuiles selon le nombre d'observations à l'intérieur
#     TuilesOverlay=TuilesOverlay.sort_values(['Id','Name'],ascending=False).groupby("Id").last() #Trie selon l'ordre et prend le premier
#     # TuilesOverlay=TuilesOverlay.groupby("Id").first()
#     return list("T"+np.unique(TuilesOverlay['Name']))