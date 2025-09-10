# -*- coding: utf-8 -*-

import click
from fordead.cli.utils import empty_to_none
from fordead.import_data import TileInfo, import_dieback_data, import_binary_raster, import_soil_data, import_stress_data, import_stress_index, import_coeff_model, import_first_detection_date_index
from fordead.writing_data import vectorizing_confidence_class, get_bins, convert_dateindex_to_datenumber, get_periodic_results_as_shapefile, get_state_at_date, union_confidence_class, write_tif
from fordead.stac.stac_module import get_tile_collection
from fordead.reflectance_extraction import extract_raster_values
from fordead.model_vegetation_index import prediction_vegetation_index
import numpy as np
import geopandas as gp
import pandas as pd
from path import Path
import warnings
import xarray as xr
import re

warnings.filterwarnings("ignore", message="`unary_union` returned None due to all-None GeoSeries. In future, `unary_union` will return 'GEOMETRYCOLLECTION EMPTY' instead.")

@click.command(name='export_results')
@click.option("-o", "--data_directory",  type=str, help="Path of the output directory", show_default=True)
@click.option("--start_date",  type=str, default='2015-06-23',
                    help="Start date for exporting results", show_default=True)
@click.option("--end_date",  type=str, default="2030-01-02",
                    help="End date for exporting results", show_default=True)
@click.option("--frequency",  type=str, default='M',
                    help="Frequency used to aggregate results, if value is 'sentinel', then periods correspond to the period between sentinel dates used in the detection, or it can be the frequency as used in pandas.date_range. e.g. 'M' (monthly), '3M' (three months), '15D' (fifteen days)", show_default=True)
@click.option("--multiple_files",  is_flag=True,
                    help="If True, one shapefile is exported for each period containing the areas in dieback at the end of the period. Else, a single shapefile is exported containing diebackd areas associated with the period of dieback", show_default=True)
@click.option("-t", "--conf_threshold_list", type = float, multiple=True, default = None, help = "List of thresholds used as bins to discretize the confidence index into several classes", show_default=True)
@click.option("-c", "--conf_classes_list", type = str, multiple=True, default = None, help = "List of classes names, if conf_threshold_list has n values, conf_classes_list must have n+1 values", show_default=True)
def cli_export_results(**kwargs):
    """
    Export results to a vectorized shapefile format.
    \f

    """
    empty_to_none(kwargs, "conf_threshold_list")
    empty_to_none(kwargs, "conf_classes_list")
    export_results(**kwargs)



def export_results(
    data_directory,
    start_date = '2015-06-23',
    end_date = "2030-01-02",
    frequency = 'M',
    multiple_files = False,
    conf_threshold_list = None,
    conf_classes_list = None,
    overwrite = False
    ):
    """
    Writes results in the chosen period, form and using chosen frequency.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/english/05_export_results/
    \f

    Parameters
    ----------
    data_directory : str
        Path of the output directory
    start_date : str
        Start date for exporting results (format : 'YYYY-MM-DD')
    end_date : str
        End date for exporting results (format : 'YYYY-MM-DD')
    frequency : str
        Frequency used to aggregate results, if value is 'sentinel', then periods correspond to the period between sentinel dates used in the detection, or it can be the frequency as used in pandas.date_range. e.g. 'M' (monthly), '3M' (three months), '15D' (fifteen days)
    multiple_files : bool
        If True, one shapefile is exported for each period containing the areas in dieback at the end of the period. Else, a single shapefile is exported containing diebackd areas associated with the period of dieback
    conf_threshold_list : list
        List of thresholds used as bins to discretize the confidence index into several classes
    conf_classes_list : list
        List of classes names, if conf_threshold_list has n values, conf_classes_list must have n+1 values
    
    Returns
    -------

    """    
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    dieback_data = import_dieback_data(tile.paths, chunks= None)
    tile.add_parameters({"start_date" : start_date,"end_date" : end_date, "frequency" : frequency, "multiple_files" : multiple_files, "conf_threshold_list": conf_threshold_list, "conf_classes_list" : conf_classes_list})
    if tile.parameters["Overwrite"] or overwrite:
        tile.delete_dirs("periodic_results_dieback","result_files") #Deleting previous detection results if they exist
        tile.delete_attributes("last_date_export")
        
    tile.add_path("confidence_index", tile.data_directory / "Results" / "confidence_index.tif")
    # tile.add_path("stress_periods", tile.data_directory / "Results" / "stress_periods.shp")
    tile.add_path("stress_periods", tile.data_directory / "Results" / "stress_periods.shp")

    exporting = (tile.dates[-1] != tile.last_date_export) if hasattr(tile, "last_date_export") else True
    # exporting = True
    if exporting:
        print("Exporting results")
        bins_as_date, bins_as_datenumber = get_bins(start_date,end_date,frequency,tile.dates)
        first_date_number = convert_dateindex_to_datenumber(dieback_data.first_date,dieback_data.state, tile.dates)
        confirmation_date = convert_dateindex_to_datenumber(dieback_data.confirmation_date,dieback_data.state, tile.dates)
        print("YOAN", confirmation_date)
        if tile.parameters["soil_detection"]:
            soil_data = import_soil_data(tile.paths, chunks= 1280)
            first_date_number_soil = convert_dateindex_to_datenumber(soil_data.first_date, soil_data.state, tile.dates)
            

        forest_mask = import_binary_raster(tile.paths["forest_mask"], chunks= 1280)
        sufficient_coverage_mask = import_binary_raster(tile.paths["sufficient_coverage_mask"], chunks= 1280)
        if tile.parameters["stress_index_mode"] is not None:
            too_many_stress_periods_mask = import_binary_raster(tile.paths["too_many_stress_periods_mask"], chunks= 1280)
            relevant_area = forest_mask & sufficient_coverage_mask & too_many_stress_periods_mask
        else:
            relevant_area = forest_mask & sufficient_coverage_mask
      
        #EXPORTING STRESS RESULTS
        if tile.parameters["stress_index_mode"] is not None:
            if conf_threshold_list is None or conf_classes_list is None or len(conf_threshold_list) == 0 or len(conf_classes_list) == 0:
                print("Parameters conf_threshold_list and conf_classes_list are not provided, stress results can't be exported")
            else:
                stress_index = import_stress_index(tile.paths["stress_index"])
                stress_data = import_stress_data(tile.paths)
                stress_list = []
                for period in range(tile.parameters["max_nb_stress_periods"]):
                    stress_period = stress_index.isel(period = period)
                    conf_threshold_list = np.array(conf_threshold_list).astype(float)
                    stress_class = vectorizing_confidence_class(stress_period, stress_data.nb_dates.isel(period = period), (relevant_area & (period < stress_data["nb_periods"])).compute(), conf_threshold_list, np.array(conf_classes_list), tile.raster_meta["attrs"])
                    if stress_class.size != 0:
                        stress_start_date = stress_data["date"].isel(change = period*2)
                        stress_start_date_number = convert_dateindex_to_datenumber(stress_start_date, period < stress_data["nb_periods"], tile.dates)
                        vector_stress_start = get_periodic_results_as_shapefile(stress_start_date_number, bins_as_date, bins_as_datenumber, relevant_area, forest_mask.attrs)
                        # stress_class.to_file(tile.paths["stress_periods"] / ("stress_class" + str(period+1) + ".shp"))
                        # vector_stress_start.to_file(tile.paths["stress_periods"] / ("stress_period" + str(period+1) + ".shp"))
                        stress_list += [gp.overlay(vector_stress_start, stress_class, how='intersection',keep_geom_type = True)]
                # for period_date in pd.unique(vector_stress_start["period"]):
                #     for class_stress in pd.unique(stress_class["class"]):
                #         stress_list += [gp.overlay(vector_stress_start[vector_stress_start["period"]==period_date], stress_class[stress_class["class"] == class_stress], how='intersection',keep_geom_type = True)]
                
                if len(stress_list) != 0:
                    stress_total = gp.GeoDataFrame( pd.concat(stress_list, ignore_index=True), crs=stress_list[0].crs)
                    stress_total.to_file(tile.paths["stress_periods"])
                else:
                    print("No stress periods detected")

        if multiple_files:
            tile.add_dirpath("result_files", tile.data_directory / "Results")
            for date_bin_index, date_bin in enumerate(bins_as_date):
                state_code = first_date_number <= bins_as_datenumber[date_bin_index]
                if tile.parameters["soil_detection"]:
                    state_code = state_code + 2*(first_date_number_soil <= bins_as_datenumber[date_bin_index])
                period_end_results = get_state_at_date(state_code,relevant_area,dieback_data.state.attrs)
                if not(period_end_results.empty):
                    period_end_results.to_file(tile.paths["result_files"] / (date_bin.strftime('%Y-%m-%d')+".shp"))
                    
        else:
            #EXPORTING DIEBACK 
            tile.add_path("periodic_results_dieback", tile.data_directory / "Results" / "periodic_results_dieback.shp")
            periodic_results = get_periodic_results_as_shapefile(first_date_number, bins_as_date, bins_as_datenumber, relevant_area, dieback_data.state.attrs)
          
            if conf_threshold_list is not None and conf_classes_list is not None and len(conf_threshold_list) != 0 and len(conf_classes_list) != 0:
                if tile.parameters["stress_index_mode"] is None:
                    print("Confidence index was not saved, parameters conf_threshold_list and conf_classes_list are ignored. Change stress_index_mode parameter in step 3 to compute confidence index")
                else:
                    stress_data = import_stress_data(tile.paths)
                    stress_index = import_stress_index(tile.paths["stress_index"])
                    confidence_area = relevant_area & dieback_data["state"] & ~soil_data["state"] if tile.parameters["soil_detection"] else relevant_area & dieback_data["state"]
               
                    confidence_index = stress_index.sel(period = (stress_data["nb_periods"]+1).where(stress_data["nb_periods"]<=tile.parameters["max_nb_stress_periods"],tile.parameters["max_nb_stress_periods"])) #The selection probably makes no sense for pixels with nb_periods higher that max_nb_stress_periods, but it doesn't matter since they are excluded from result exports, but it removes bugs of inexistant period values.
                    nb_dates = stress_data["nb_dates"].sel(period = (stress_data["nb_periods"]+1).where(stress_data["nb_periods"]<=tile.parameters["max_nb_stress_periods"],tile.parameters["max_nb_stress_periods"]))  #The selection probably makes no sense for pixels with nb_periods higher that max_nb_stress_periods, but it doesn't matter since they are excluded from result exports, but it removes bugs of inexistant period values.
                   
                    write_tif(confidence_index.where(confidence_area,0), forest_mask.attrs,nodata = 0, path = tile.paths["confidence_index"])
                   
                    confidence_class = vectorizing_confidence_class(confidence_index, nb_dates, confidence_area.compute(), conf_threshold_list, np.array(conf_classes_list), tile.raster_meta["attrs"])

                    periodic_results = union_confidence_class(periodic_results, confidence_class)
                    
            if not(periodic_results.empty):
                periodic_results.to_file(tile.paths["periodic_results_dieback"],index = None)
            del periodic_results
            
            #EXPORTING SOIL DETECTION
            if tile.parameters["soil_detection"]:
                tile.add_path("periodic_results_soil", tile.data_directory / "Results" / "periodic_results_soil.shp")
                periodic_results = get_periodic_results_as_shapefile(first_date_number_soil, bins_as_date, bins_as_datenumber, relevant_area, soil_data.state.attrs)
                if not(periodic_results.empty):
                    periodic_results.to_file(tile.paths["periodic_results_soil"])
                del periodic_results
        tile.last_date_export = tile.dates[-1]
        tile.save_info()
        dieback_data.close()
    else:
        print("Results already exported")

def extract_results(data_directory, points, output_dir=None,
                    index_name = "id_point", chunks = 100):
    
    """
    Extract points of interest from the results computed at the MGRS-tile scale.

    Parameters
    ----------
    data_directory : str
        Path of the directory containing results from the region of interest
    points: str | geopandas.GeoDataFrame
        Path to vector file containing points to be extracted or geodataframe containing points to be extracted
    output_dir: str
        Path of the directory where results should be exported. If None, results are returned.
    index_name: str
        Name of the column containing the point ID. If it does not exist, it is added filling it with the index values of the points.
    chunks: int
        Chunk length to import data as dask arrays and save RAM, advised if computed area in data_directory is large

    Returns
    -------
    None or Tuple[pandas.DataFrame]
        If output_dir is None, returns the timeseries, current_state and periods dataframes
    
    Details
    -------
    Extracted results:
        - timeseries
            - vi: vegetation index, 
            - masks: the combination of all masks
            - predicted_vi: predicted vegetation index
            - diff_vi: difference between vegetation index
            - anomaly: anomaly of vi

        - current_state: current state variables at the last date of analysis (dieback, soil and other timeless masks)
            - dieback_: the state of dieback detection variables; 
              Note that dieback_first_date is the start date period, i.e. "healthy" if dieback_state=0 or "dieback" if dieback_state=1.
            - soil_: the state of soil detection variables, same note as above
        
        - periods.csv: dieback and healthy period summary
            - period: the period index
            - cum_diff: cumulated difference between vegetation index and predicted vegetation index
            - start_date: starting date of the "dieback" or "stress" period, i.e. the first date of a series of 3+ anmalies
            - end_date: actually the starting date of the following healthy period, i.e. the first date of a series of 3+ non-anomalies
            - start_date_index, end_date_index: date indexes of the start and end dates
            - nb_dates: number of dates in the "dieback" or "stress" period,
              starting at the first unconfirmed anomaly (i.e. usually 2 dates before the first confirmed anomaly),
              and ending at the date before the first confirmed non-anomaly
              This is why usually nb_dates = (end_date_index - start_date_index)+2
        
        - static: static data of the tile
            - coeff_i: coefficients of the model
            - forest_mask: the forest mask value
            - first_detection_date: the starting date of detection of the pixel timeseries
            - first_detection_date_index: the date index of first_detection_date
    """
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    
    # IMPORTING ALL DATA

    if isinstance(points, str):
        points = gp.read_file(points)

    if not isinstance(points, gp.GeoDataFrame):    
        raise TypeError(
            "points must be a path to a vector file or a " \
            "geopandas GeoDataFrame of points"
        )
    
    if (points.geometry.geom_type != "Point").any():
        raise ValueError("All points must be of type Point")
    
    if index_name not in points.columns:
        print(f"Column '{index_name}' not found in points. " \
              f"Creating '{index_name}' with index values.")
        points = points.reset_index().rename(columns={"index": index_name})
    
    if points[index_name].duplicated().any():
        raise ValueError(f"Duplicates found in column '{index_name}', points IDs must be unique: choose another name for 'index_name'.")

    ts_col = get_tile_collection(tile)
    ts = extract_raster_values(ts_col, points, bands_to_extract=None, chunksize=chunks, by_chunk=True, dropna=False, dtype=None)
    ts.rename(columns={"Anomalies": "anomaly", "VegetationIndex": "vi", "Masks": "masks"}, inplace=True)
    binary_keys = [k for k in ["anomaly", "masks"] if k in ts]
    if len(binary_keys) > 0:
        ts.loc[:,binary_keys] = ts.loc[:,binary_keys].fillna(0)
        ts = ts.astype({k: bool for k in binary_keys})
        
    coeff_model = import_coeff_model(tile.paths["coeff_model"],chunks = chunks)
    
    pred = prediction_vegetation_index(coeff_model,ts.Date.drop_duplicates().to_list()).rename({"Time":"time", "coeff":"band"})
    pred.name = "value"
    pred["time"] = pd.to_datetime(pred.time)
    pred["band"] = "predicted_vi"
    p = extract_raster_values(pred, points, bands_to_extract=None, chunksize=chunks, by_chunk=True, dropna=False, dtype=None)
    ts = ts.merge(p, "left", on=["Date", index_name])
    ts["diff_vi"] = ts.loc[:, "vi"] - ts.loc[:, "predicted_vi"]


    dieback_data = import_dieback_data(tile.paths,chunks = chunks)
    current_data = dieback_data.rename({k: f"dieback_{k}" for k in list(dieback_data)})
    if tile.parameters["soil_detection"]:
        soil_data = import_soil_data(tile.paths,chunks = chunks)
        soil_data = soil_data.rename({k: f"soil_{k}" for k in list(soil_data)})
        current_data = xr.merge([current_data, soil_data], join='outer')
    
    current_data = current_data.assign_coords(time=pred.time.values[-1])
    current_data = current_data.to_dataarray("band")
    current_data.name="value"
    current_state = extract_raster_values(current_data, points, bands_to_extract=None, chunksize=chunks, by_chunk=False, dropna=False, dtype=None)
    
    for k in current_state:
        if k.endswith("_date") or k.endswith("_date_unconfirmed"):
            current_state.rename(columns={k: f"{k}_index"}, inplace=True)
            current_state[k] = tile.dates[current_state[f"{k}_index"]]
        elif k.endswith("_date_index"):
            current_state[re.sub("_index", "", k)] = tile.dates[current_state[k]]


    static_data = import_first_detection_date_index(tile.paths["first_detection_date_index"],chunks = chunks).rename("first_detection_date_index").to_dataset()
    if "forest_mask" in tile.paths:
        forest_mask = import_binary_raster(tile.paths["forest_mask"],chunks = chunks)
        static_data["forest_mask"] = forest_mask
    coeff_model["coeff"] = [f"coeff_{i}" for i in coeff_model.coeff.values]
    coeff_model = coeff_model.rename({"coeff": "band"}).to_dataset("band")
    static_data = xr.merge([static_data, coeff_model], join='outer')
    static_data = static_data.assign_coords(time=pred.time.values[-1])
    static_data = static_data.to_dataarray("band")
    static_data.name="value"
    static_df = extract_raster_values(static_data, points, bands_to_extract=None, chunksize=chunks, by_chunk=False, dropna=False, dtype=None)
    static_df = static_df.drop(columns = "Date")
    static_df["first_detection_date_index"] = static_df["first_detection_date_index"].astype(int)
    static_df["first_detection_date"] = tile.dates[static_df["first_detection_date_index"]]

    # static_data["first_detection_date_index"] = first_detection_date_index


    stress_df = None
    if tile.parameters["stress_index_mode"] is not None:
        stress_data = import_stress_data(tile.paths,chunks = chunks)
        stress_list = []
        for point_index in range(len(points)):
            x = points.geometry.x[point_index]
            y = points.geometry.y[point_index]
            stress = stress_data.sel(x=x, y=y, method="nearest")
            stress_periods = stress[["cum_diff", "nb_dates"]].to_dataframe().drop(columns = ["spatial_ref", "band"])
            start_dates = stress["date"].values[np.arange(0, len(stress["date"]), 2)]
            end_dates = stress["date"].values[np.arange(1, len(stress["date"]), 2)]
            stress_periods["start_date_index"] = start_dates
            stress_periods["end_date_index"] = np.append(end_dates, 0)
            stress_periods = stress_periods.query("start_date_index!=0").copy()
            stress_periods["start_date"] = tile.dates[stress_periods["start_date_index"]]
            stress_periods["end_date"] = tile.dates[stress_periods["end_date_index"]]
            stress = stress_periods
            if stress is not None:
                stress.loc[:, index_name] = points.iloc[point_index].loc[index_name]
                stress["id_pixel"] = point_index
                stress_list.append(stress.reset_index(drop = False))
        if len(stress_list) > 0 :
            stress_df = pd.concat(stress_list, ignore_index=True)

    if output_dir is None:
        return ts, current_state, stress_df, static_df
    
    ts.to_csv(output_dir/"timeseries.csv", index = False, header = True)
    current_state.to_csv(output_dir/"current_state.csv", index = False, header = True)
    if stress_df is not None:
        stress_df.to_csv(output_dir/"periods.csv", index = False, header = True)
    static_df.to_csv(output_dir/"static.csv", index = False, header = True)
