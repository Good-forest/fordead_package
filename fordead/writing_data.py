# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:32:26 2020

@author: Raphael Dutrieux
"""
import rioxarray
from numpy import uint8
import pandas as pd
import datetime
import numpy as np
import xarray as xr
import rasterio
from affine import Affine
import geopandas as gp
import dask.array as da
from fordead.model_vegetation_index import prediction_vegetation_index
from fordead.masking_vi import get_dict_vi
from scipy import ndimage
import json
from fordead.import_data import import_stress_index, import_coeff_model, import_dieback_data, import_masked_vi, import_first_detection_date_index, TileInfo, import_binary_raster, import_soil_data,import_resampled_sen_stack, import_stress_data

def write_raster(data_array, path, compress_vi):
        
    # dem = data_array.to_dataset(name="dem")
    # encoding = {"dem": {'zlib': True, "dtype" : "int16", "scale_factor" : 0.001, "_FillValue" : 0}}
    # dem.to_netcdf(path, encoding=encoding)
    if compress_vi:
        data_array.encoding["dtype"]="int16"
        data_array.encoding["scale_factor"]=0.001
        data_array.encoding["_FillValue"]=-1
    if str(path).endswith(".nc"):
        tiled = False
    else:
        tiled = True
    data_array.rio.to_raster(path, windowed = False, tiled = tiled)

def write_tif(data_array, attributes, path, nodata = None):
    """
    Writes raster to the disk

    Parameters
    ----------
    data_array : xarray DataArray
        Object to be written
    attributes : dict
        Dictionnary containing attributes used to write the data_array ("crs","nodata","scales","offsets")
    path : str
        Path of the file to which data will be written
    nodata : int or float, optional
        Number used as nodata. If None, the nodata attribute of the object will be kept. The default is None.

    Returns
    -------
    None.

    """

        
    data_array.attrs=attributes
    # data_array.rio.crs=data_array.crs.replace("+init=","") #Remove "+init=" which it deprecated

    args={}
    if data_array.dtype==bool: #Bool rasters can't be written, so they have to be converted to int8, but they can still be written in one bit with the argument nbits = 1
        data_array=data_array.astype(uint8)
        args["nbits"] = 1
    if nodata != None:
        data_array.attrs["_FillValue"]=nodata
        data_array.attrs["nodata"]=nodata
        
    if len(data_array.dims)>=3: #If data_array has several bands
        for dim in data_array.dims:
            if dim != "x" and dim != "y":
                data_array=data_array.transpose(dim, 'y', 'x') #dimension which is not x or y must be first
        # data_array.attrs["scales"]=data_array.attrs["scales"]*data_array.shape[0]
        # data_array.attrs["offsets"]=data_array.attrs["offsets"]*data_array.shape[0]
    
    # data_array.attrs["nodatavals"]=(0,)
    # data_array.attrs["scales"]=(0,)
    # data_array.attrs["offsets"]=(0,)

    data_array.rio.to_raster(path,windowed = False, **args, tiled = True)



def get_bins(start_date,end_date,frequency,dates):
    """
    Creates bins from the start_date (or first used SENTINEL date if it is later than the start date) to the end_date (or last used SENTINEL date if it is earlier than the end_date) with specified frequency

    Parameters
    ----------
    start_date : str
        Date in the format 'YYYY-MM-DD'
    end_date : str
        Date in the format 'YYYY-MM-DD'
    frequency : str
        Frequency as used in pandas.date_range (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html), e.g. 'M' (monthly), '3M' (three months), '15D' (fifteen days). It can also be 'sentinel', then bins correspond to the list given as parameter 'dates'
    dates : list
        List of dates used for detection

    Returns
    -------
    bins_as_date : numpy array
        Bins as an array of dates in the format 'YYYY-MM-DD'
    bins_as_datenumber : numpy array
        Bins as an array of integers corresponding to the number of days since "2015-01-01"

    """
    
    if frequency == "sentinel":
        bins_as_date = pd.DatetimeIndex(dates)
    else:
        bins_as_date=pd.date_range(start=start_date, end = end_date, freq=frequency)

    # bins_as_date = bins_as_date.insert(0,datetime.datetime.strptime(start_date, '%Y-%m-%d'))
    # bins_as_date = bins_as_date.insert(len(bins_as_date),datetime.datetime.strptime(end_date, '%Y-%m-%d'))
    bins_as_datenumber = (bins_as_date-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days  
    
    bin_min = max((datetime.datetime.strptime(start_date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days, (datetime.datetime.strptime(dates[0], '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days)
    bin_max = min((datetime.datetime.strptime(end_date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days, (datetime.datetime.strptime(dates[-1], '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days)

    kept_bins = ndimage.binary_dilation(np.logical_and(bins_as_datenumber>=bin_min,bins_as_datenumber<=bin_max),iterations=1,structure=np.array([False,True,True])) #Kept bins are those within bin_min and bin_max plus one bin after bin_max

    bins_as_date = bins_as_date[kept_bins]
    bins_as_datenumber = bins_as_datenumber[kept_bins]
    
    return bins_as_date, bins_as_datenumber

def convert_dateindex_to_datenumber(date_array, mask_array, dates):
    """
    Converts array containing dates as an index to an array containing dates as the number of days since "2015-01-01" or to a no data value if masked

    Parameters
    ----------
    date_array : xarray DataArray 
        array containing date indices
    mask_array : xarray DataArray 
        mask array, pixels containing False are given no data value of 99999999
    dates : array
        Array of dates in the format "YYYY-MM-DD", index of the date in this array corresponds to the indices in date_array.

    Returns
    -------
    results_date_number : xarray DataArray
        DataArray with dates as the number of days since "2015-01-01", or no data value of 99999999

    """
    
    used_dates_numbers = (pd.to_datetime(dates)-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days
    results_date_number = used_dates_numbers[date_array.data.ravel()]
    results_date_number = np.reshape(np.array(results_date_number),date_array.shape)
    results_date_number[~mask_array.data] = 99999999
    
    return results_date_number


def get_periodic_results_as_shapefile(first_date_number, bins_as_date, bins_as_datenumber, relevant_area, attrs):
    """
    Aggregates pixels in array containing dates, based on the period they fall into, then vectorizes results masking dates oustide the bins.

    Parameters
    ----------
    first_date_number : array
        Array containing dates as the number of days since "2015-01-01"
    bins_as_date : array
        Array containing dates used as bins in the format "YYYY-MM-JJ"
    bins_as_datenumber : array
        Array containing dates used as bins, as the number of days since "2015-01-01" 
    relevant_area : array
        Mask where pixels with value False will be ignored.
    attrs : dict
        Dictionnary containing 'tranform' and 'crs' to create the vector.

    Returns
    -------
    period_end_results : geopandas dataframe
        Polygons containing the period during which dates fall.

    """
    
    inds_soil = np.digitize(first_date_number, bins_as_datenumber, right = True)
    # geoms_period_index = list(
    #             {'properties': {'period_index': v}, 'geometry': s}
    #             for i, (s, v) 
    #             in enumerate(
    #                 rasterio.features.shapes(inds_soil.astype("uint16"), mask =  (relevant_area & (inds_soil!=0) &  (inds_soil!=len(bins_as_date))).data , transform=Affine(*attrs["transform"]))))
    geoms_period_index = list(
                {'properties': {'period_index': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    rasterio.features.shapes(inds_soil.astype("uint16"), mask =  (relevant_area & (inds_soil!=0) & (inds_soil!=len(bins_as_date))).compute().data , transform=relevant_area.rio.transform()))) #Affine(*attrs["transform"])
    gp_results = gp.GeoDataFrame.from_features(geoms_period_index)

    if gp_results.size != 0:
        gp_results.period_index=gp_results.period_index.astype(int)
            #If you want to reactivate start and end columns
        # gp_results.insert(0,"start",(bins_as_date[gp_results.period_index-1] + pd.DateOffset(1)).strftime('%Y-%m-%d'))
        # gp_results.insert(1,"end",(bins_as_date[gp_results.period_index]).strftime('%Y-%m-%d'))
        # gp_results.insert(0,"period", (gp_results["start"] + " - " + gp_results["end"]))
            #If you only want period column
        gp_results.insert(0,"period", ((bins_as_date[gp_results.period_index-1] + pd.DateOffset(1)).strftime('%Y-%m-%d') + " - " + (bins_as_date[gp_results.period_index]).strftime('%Y-%m-%d')))
        ###############
        gp_results.crs = relevant_area.rio.crs #attrs["crs"].replace("+init=","")
        gp_results = gp_results.drop(columns=['period_index'])
    else:
        print("No detection in this area")

    return gp_results

def get_state_at_date(state_code,relevant_area,attrs):
    """
    Vectorizes array 'state_code' using 'relevant_area' as mask

    Parameters
    ----------
    state_code : array
        Array to vectorize, in which value 0 is ignored, 1 is 'Anomaly', 2 is "Bare ground" and 3 is 'Bare ground after anomaly'
    relevant_area : array
        Mask where pixels with value False will be ignored.
    attrs : dict
        Dictionnary containing 'tranform' and 'crs' to create the vector.

    Returns
    -------
    period_end_results : geopandas dataframe
        Polygons from vectorization of state_code, aggregated according to value 'Anomaly', "Bare ground" and 'Bare ground after anomaly'

    """
    
    geoms = list(
                {'properties': {'state': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    rasterio.features.shapes(state_code.astype("uint8"), mask =  np.logical_and(relevant_area.data,state_code!=0).compute(), transform=Affine(*attrs["transform"]))))
    period_end_results = gp.GeoDataFrame.from_features(geoms)
    
    period_end_results = period_end_results.replace([1, 2, 3], ["Anomaly","Bare ground","Bare ground after anomaly"])
    
    period_end_results.crs = attrs["crs"].replace("+init=","")
    return period_end_results

def vectorizing_confidence_class(confidence_index, nb_dates, relevant_area, bins_classes, classes, attrs):
    """
    Classifies pixels in the relevant area into dieback classes based on the confidence index and the number of unmasked dates since the first anomaly. 
    
    Parameters
    ----------
    confidence_index : xarray DataArray (x,y) (float)
        Confidence index.
    nb_dates : xarray DataArray (x,y) (int)
        number of unmasked dates since the first anomaly.
    relevant_area : xarray DataArray (x,y) (bool)
        Array with True where pixels will be vectorized, and False where ignored.
    bins_classes : list of float
        List of bins to classify pixels based on confidence_index
    classes : list of str
        List with names of classes (length of classes must be length of bins_classes + 1)
    attrs : dict
        Dictionnary containing 'tranform' and 'crs' to create the vector.

    Returns
    -------
    gp_results : geopandas geodataframe
        Polygons from pixels in the relevant areacontaining the dieback class.

    """

    digitized = np.digitize(confidence_index, bins_classes)  
    digitized[nb_dates.data==3]=0
    geoms_class = list(
                {'properties': {'class_index': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    rasterio.features.shapes(digitized.astype(np.uint8), 
                                             mask = relevant_area.data, 
                                             transform=confidence_index.rio.transform())))
    
    gp_results = gp.GeoDataFrame.from_features(geoms_class)

    if gp_results.size != 0:
        gp_results.class_index=gp_results.class_index.astype(int)
        gp_results.insert(1,"class",classes[gp_results.class_index])
        gp_results.crs = confidence_index.rio.crs
        gp_results = gp_results.drop(columns=['class_index'])
    return gp_results


def union_confidence_class(periodic_results, confidence_class):
    """
    Computes union between periodic_results containing the dates of detection, and confidence_class containing the confidence classes
    Polygons not intersecting correspond to areas detected as bare ground and their class column is filled with "Bare ground"

    Parameters
    ----------
    period_end_results : geopandas dataframe
        Polygons where dieback is detected, containing the period of the first anomaly detected
    confidence_class : geopandas dataframe
        Polygons containing the confidence class

    Returns
    -------
    union : geopandas dataframe
        Union of periodic_results and confidence_class

    """
    
    
    # confidence_class = gp.read_file(path_confidence_class)
    if periodic_results.size != 0:
        union = gp.overlay(periodic_results, confidence_class, how='union',keep_geom_type = True)
        union = union.dropna(subset=['period'])
        union = union.fillna("Bare ground")
        
        return union
    else:
        return periodic_results

    # union = union.explode()
    # union = union[union.geom_type != 'Point']
    # union = union[union.geom_type != 'MultiLineString']
    # union = union[union.geom_type != 'LineString']
    # union = union[union.geom_type != 'MultiPoint']
    
def get_rasterized_validation_data(ground_obs_path, raster_metadata, ground_obs_buffer = None, name_column  = "id"):

        #Rasterize données terrain
    ground_obs = gp.read_file(ground_obs_path)
    ground_obs=ground_obs.to_crs(crs=raster_metadata["attrs"]["crs"])
            
    all_touched = True if np.all(ground_obs.geom_type == 'Point') else False           
    
    if ground_obs_buffer is not None:
        ground_obs['geometry']=ground_obs.geometry.buffer(ground_obs_buffer)
        ground_obs=ground_obs[~(ground_obs.is_empty)]
        # ground_obs=ground_obs[ground_obs["IndSur"]==1]
    
    ground_obs[name_column]= ground_obs[name_column].astype(int)
    
    ground_obs_json_str = ground_obs.to_json()
    ground_obs_json_dict = json.loads(ground_obs_json_str)
    ground_obs_jsonMask = [(feature["geometry"],feature["properties"][name_column]) for feature in ground_obs_json_dict["features"]]
    
    rasterized_validation_data = rasterio.features.rasterize(ground_obs_jsonMask,
                                                              out_shape = (raster_metadata["sizes"]["y"],
                                                                          raster_metadata["sizes"]["x"]),
                                                              dtype="int32",
                                                              transform=raster_metadata["attrs"]['transform'],
                                                              all_touched=all_touched)  
    
    return xr.DataArray(rasterized_validation_data,coords={"y" : raster_metadata["coords"]["y"],"x" : raster_metadata["coords"]["x"]}, dims=["y","x"])


def export_csv(
    data_directory,
    ground_obs_path,
    ground_obs_buffer = None,
    name_column = "id"):
    """
    Produce the anomaly detection from the model, along with exporting as two .csv files data relating to pixels within the ground_obs_path shapefile.
    \f
    Parameters
    ----------
    data_directory: str
        Path of the output directory
    ground_obs_path: str
        Path to the shapefile containing ground observation polygons
    ground_obs_buffer: int
        Buffer applied to vectors, positive for dilation, negative for erosion.
    name_column: str
        Name of the column containing the ID of the ground observation polygon

    """

    # data_directory ="E:/fordead/Data/Feuillu/output_crswir"
    # ground_obs_path = "E:/fordead/Data/Feuillu/hetre/plac_hetre_2020.shp"
    # name_column = "N_plac"


    tile = TileInfo(data_directory)
    tile = tile.import_info()
    tile.add_dirpath("validation", tile.data_directory / "Validation")
    
    forest_mask = import_binary_raster(tile.paths["forest_mask"])
    forest_mask = import_binary_raster(tile.paths["forest_mask"], chunks= 1280)
    sufficient_coverage_mask = import_binary_raster(tile.paths["sufficient_coverage_mask"], chunks= 1280)
    if tile.parameters["stress_index_mode"] is not None:
        too_many_stress_periods_mask = import_binary_raster(tile.paths["too_many_stress_periods_mask"], chunks= 1280)
        relevant_area = forest_mask & sufficient_coverage_mask & too_many_stress_periods_mask
    else:
        relevant_area = forest_mask & sufficient_coverage_mask
    # if type(ground_obs) is str: ground_obs = gp.read_file(ground_obs)

    raster_id_validation_data=get_rasterized_validation_data(ground_obs_path, tile.raster_meta, ground_obs_buffer, name_column)
    # raster_binary_validation_data = (raster_id_validation_data!=0) & relevant_area
    raster_binary_validation_data = (raster_id_validation_data!=0)

    nb_pixels = int(np.sum(raster_binary_validation_data))
    
    # raster_binary_validation_data.plot()
    
    if nb_pixels != 0:
        #Verify if there are new SENTINEL dates
            first_detection_date_index = import_first_detection_date_index(tile.paths["first_detection_date_index"])
            coeff_model = import_coeff_model(tile.paths["coeff_model"])
            dieback_data = import_dieback_data(tile.paths)
            stress_data = import_stress_data(tile.paths)
            # stress_index = import_stress_index(tile.paths["stress_index"])

            
            # #DECLINE DETECTION
            # for date_index, date in enumerate(tile.dates):
            #     masked_vi = import_masked_vi(tile.paths, date)
            #     if tile.parameters["correct_vi"]:
            #         masked_vi["vegetation_index"], tile.correction_vi = correct_vi_date(masked_vi,forest_mask, tile.large_scale_model, date, tile.correction_vi)

                
            #     dieback = (dieback_data["first_date"] <= date_index) & dieback_data["state"]
            #     # predicted_vi=
            #     B2 = import_resampled_sen_stack(tile.paths["Sentinel"][date], ["B2"], interpolation_order = 0, extent = tile.raster_meta["extent"])

            #     dict_date = {'IdZone': raster_id_validation_data.data[raster_binary_validation_data],
            #           "IdPixel" : range(nb_pixels),
            #           "Date" : [date]*nb_pixels,
            #           "vi" : (masked_vi["vegetation_index"].data)[raster_binary_validation_data],
            #           "Mask" : (masked_vi["mask"].data)[raster_binary_validation_data],
            #           "Predicted_vi" : prediction_vegetation_index(coeff_model,[date]).squeeze("Time").data[raster_binary_validation_data]}
             
            #     if date >= tile.parameters["min_last_date_training"]:
            #         dict_date.update({"Anomaly" : import_binary_raster(tile.paths["AnomaliesDir"] / str("Anomalies_" + date + ".tif")).data[raster_binary_validation_data]})
            #     else:
            #         dict_date.update({"Anomaly" : [False]*nb_pixels})
                    
            #     del masked_vi
            #     Results = pd.DataFrame(data=dict_date)
            #     Results.to_csv(tile.paths["validation"] / 'Evolution_data.csv', mode='a', index=False,header=not((tile.paths["validation"] / 'Evolution_data.csv').exists()))
            #     print('\r', date, " | ", len(tile.dates)-date_index-1, " remaining      ", sep='', end='', flush=True) if date_index != (len(tile.dates) -1) else print('\r', "                                              ", sep='', end='\r', flush=True) 

            data = raster_id_validation_data.to_dataframe(name="IdZone")
            data = data[data.IdZone!=0]
            
            dict_results = {'IdZone': raster_id_validation_data.data[raster_binary_validation_data],
                  "IdPixel" : range(nb_pixels),
                  "x" : data.index.get_level_values('x'),
                  "y" : data.index.get_level_values('y')}
            dict_results.update(
                {"coeff"+str(i) : coeff_model.sel(coeff = i).data[raster_binary_validation_data] for i in range(1,6)})
            dict_results.update(
                {"forest_mask" : forest_mask.data[raster_binary_validation_data],
                  "sufficient_coverage" : sufficient_coverage_mask.data[raster_binary_validation_data],
                  "too_many_stress_periods" : too_many_stress_periods_mask.data[raster_binary_validation_data],
                  "first_detection_date" : tile.dates[first_detection_date_index.data[raster_binary_validation_data]],
                  "dieback_state" : dieback_data["state"].data[raster_binary_validation_data],
                  "dieback_first_date" : tile.dates[dieback_data["first_date"].data[raster_binary_validation_data]],
                  "nb_stress_periods" : stress_data["nb_periods"].data[raster_binary_validation_data]})
            # dict_results.update({
            #       "stress_index"+str(i) : stress_index.sel(period = i).data[raster_binary_validation_data] for i in range(1,tile.parameters["max_nb_stress_periods"]+2)})
            dict_results.update({"vegetation_index" : tile.parameters["vi"],
                  "threshold_anomaly" : tile.parameters["threshold_anomaly"]})
                  
            
            Results = pd.DataFrame(data=dict_results)
            Results.to_csv(tile.paths["validation"] / 'Pixel_data.csv', mode='w', index=False,header=True)
            # Results2.to_csv(tile.paths["validation"] / 'Pixel_data.csv', mode='a', index=False,header=not((tile.paths["validation"] / 'Pixel_data.csv').exists()))

            tile.save_info()