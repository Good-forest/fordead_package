# -*- coding: utf-8 -*-

#%% =============================================================================
#   LIBRARIES
# =============================================================================

# %%

# import time
import click
from pathlib import Path
# import geopandas as gp
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import concurrent.futures
#%% ===========================================================================
#   IMPORT FORDEAD MODULES 
# =============================================================================
from fordead.cli.utils import empty_to_none
from fordead.import_data import TileInfo, get_band_paths, get_cloudiness, import_resampled_sen_stack, import_soil_data, initialize_soil_data, get_raster_metadata
from fordead.masking_vi import compute_masks, compute_vegetation_index, get_bands_and_formula, get_source_mask, compute_user_mask
from fordead.writing_data import write_raster, write_tif

#%% =============================================================================
#   FONCTIONS
# =============================================================================
def process_one_wrapper(args):
    return process_one(*args)

def process_mask_wrapper(args):
    return process_mask(*args)

def process_mask(tile, date, date_index, soil_data, stack_bands, sentinel_source, apply_source_mask, soil_detection, formula_mask, invalid_values):
    if soil_detection:
        mask = compute_masks(stack_bands, soil_data, date_index)
    else:
        mask = compute_user_mask(stack_bands, formula_mask)

    mask = mask | invalid_values
    if apply_source_mask:
        mask = mask | get_source_mask(tile.paths["Sentinel"][date], sentinel_source, extent = tile.raster_meta["extent"]) #Masking with source mask if option chosen

    write_tif(mask, tile.raster_meta["attrs"], tile.paths["MaskDir"] / ("Mask_"+date+".tif"),nodata=0)
    del mask, stack_bands, invalid_values


import xarray as xr
def process_one(tile, date, interpolation_order, compress_raster, compress_vi=False):
    stack_bands = import_resampled_sen_stack(tile.paths["Sentinel"][date], tile.used_bands, interpolation_order = interpolation_order, extent = tile.raster_meta["extent"])

    vegetation_index = compute_vegetation_index(stack_bands, formula = tile.vi_formula)


    invalid_values = vegetation_index.isnull() | np.isinf(vegetation_index) | np.isnan(vegetation_index)
    vegetation_index = vegetation_index.where(~invalid_values,0)

    if compress_raster:
        write_tif(vegetation_index, tile.raster_meta["attrs"],tile.paths["VegetationIndexDir"] / ("VegetationIndex_"+date+".tif"),nodata=0)
    else:
        write_raster(vegetation_index, tile.paths["VegetationIndexDir"] / ("VegetationIndex_"+date+".nc"), compress_vi)
    return date, (stack_bands, invalid_values)

def process_batch_loop(tile, new_dates, soil_data, interpolation_order, sentinel_source, apply_source_mask, soil_detection, formula_mask, compress_raster, compress_vi, progress):
    for date_index, date in enumerate(tqdm(tile.dates, disable=not progress, desc="Processing")):
        if not date in new_dates: continue
        date, (stack_bands, invalid_values) = process_one(tile, date, interpolation_order, compress_raster, compress_vi)
        process_mask(tile, date, date_index, soil_data, stack_bands, sentinel_source, apply_source_mask, soil_detection, formula_mask, invalid_values)

    return soil_data

def process_batch_multithread(tile, new_dates, soil_data, interpolation_order, sentinel_source, apply_source_mask, soil_detection, formula_mask, compress_raster, compress_vi, progress):
    vi_results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(process_one_wrapper, (tile, date, interpolation_order, compress_raster, compress_vi))
            for  date in tile.dates if date in new_dates
        ]
        for future in tqdm(as_completed(futures), total=len(futures), disable=not progress, desc="Processing"):
            vi_results.append(future.result())

    mask_values = {}
    for date, (stack_bands, invalid_values) in vi_results:
        mask_values[date] = {
            "invalid_values" : invalid_values,
            "stack_bands" : stack_bands
        }

    for date_index, date in enumerate(tqdm(tile.dates, disable=not progress, desc="Processing")):
        if date not in new_dates: continue
        process_mask_wrapper((tile, date, date_index, soil_data, mask_values[date]["stack_bands"], sentinel_source, apply_source_mask, soil_detection, formula_mask, mask_values[date]["invalid_values"]))
    return soil_data

def compute_masked_vegetationindex(
    input_directory,
    data_directory,
    start_date = "2015-01-01",
    end_date=None,
    lim_perc_cloud=0.4,
    interpolation_order = 0,
    sentinel_source = "theia",
    apply_source_mask = False,
    soil_detection = True,
    formula_mask = "(B2 >= 700)",
    vi = "CRSWIR",
    compress_vi = False,
    compress_raster = True,
    ignored_period = None,
    extent_shape_path=None,
    path_dict_vi = None,
    progress=True,
    multi_process=False
    ):
    """
    Computes masks and masked vegetation index for each SENTINEL date under a cloudiness threshold.
    Masks include shadows, clouds, soil, pixels ouside satellite swath, and the mask from SENTINEL data provider if the option is chosen.
    Results are written in the chosen directory.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/english/01_compute_masked_vegetationindex/
    
    Parameters
    ----------
    input_directory : str
        Path of the directory with Sentinel dates
    data_directory : str
        Path of the output directory
    start_date : str
        First date to process, dates before this date will be ignored. Format : 'YYYY-MM-DD'
    lim_perc_cloud : float
        Maximum cloudiness at the tile scale, used to filter used SENTINEL dates. Set parameter as -1 to not filter based on cloudiness
    interpolation_order : int
        interpolation order for bands at 20m resolution : 0 = nearest neighbour, 1 = linear, 2 = bilinéaire, 3 = cubique
    sentinel_source : str
        Source of data, can be 'theia' et 'scihub' et 'peps'
    apply_source_mask : bool
        If True, applies the mask from SENTINEL-data supplier
    soil_detection : bool
        If True, bare ground is detected and used as mask, but the process has not been tested on other data than THEIA data in France (see https://fordead.gitlab.io/fordead_package/docs/user_guides/english/01_compute_masked_vegetationindex/). If False, mask from formula_mask is applied.
    formula_mask : str
        formula whose result would be binary, as described here https://fordead.gitlab.io/fordead_package/reference/fordead/masking_vi/#compute_vegetation_index. Is only used if soil_detection is False.
    vi : str
        Chosen vegetation index
    compress_vi : bool
        If True, stores the vegetation index as low-resolution floating-point data as small integers in a netCDF file. Uses less disk space but can lead to very small difference in results as the vegetation index is rounded to three decimal places
    compress_raster : bool
        If True, compresses the output rasters using the tiff output and stdz algorithm. This can significantly reduce the file size without losing any data. Instead, classical output with netcdf (.nc) format is used.
    ignored_period : list of two strings
        Period whose Sentinel dates to ignore (format 'MM-DD', ex : ["11-01","05-01"])
    extent_shape_path : str
        Path of shapefile used as extent of detection, if None, the whole tile is used
    path_dict_vi : str
        Path of text file to add vegetation index formula, if None, only built-in vegetation indices can be used (CRSWIR, NDVI)
    progress : bool, optional
        Whether to show a progress bar. Defaults to True.
    """
    # if extent_shape_path is not None: data_directory = Path(data_directory).parent / Path(extent_shape_path).stem

    # Creation of TileInfo object. If it already exists in the specified directory, it is imported. 
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    
    # Parameters used are added to the TileInfo Object
    tile.add_parameters({
        "start_date_train" : start_date,
        "lim_perc_cloud" : lim_perc_cloud,
        "interpolation_order" : interpolation_order,
        "sentinel_source" : sentinel_source,
        "apply_source_mask" : apply_source_mask,
        "vi" : vi, "extent_shape_path" : extent_shape_path,
        "path_dict_vi" : path_dict_vi,
        "soil_detection" : soil_detection,
        "formula_mask" : formula_mask,
        "ignored_period" : ignored_period,
        "compress_vi" : compress_vi
        })
  
    # If parameters added differ from previously used parameters, all previous computation results are deleted
    if tile.parameters["Overwrite"] : 
        tile.delete_dirs("VegetationIndexDir", "MaskDir","coeff_model", "AnomaliesDir","state_dieback", "state_soil","periodic_results_dieback","result_files","timelapse","series", "nb_periods_stress")
        tile.delete_files("sufficient_coverage_mask","too_many_stress_periods_mask")
        tile.delete_attributes("last_computed_anomaly","dates","last_date_export")

    # All SENTINEL data in the input directory is detected, and paths are added to the TileInfo object. For example, after this operation tile.paths["Sentinel"]["YYYY-MM-DD"]["B2"] brings up the path to the B2 band file of the specified date?
    tile.getdict_datepaths("Sentinel",Path(input_directory), end_date) #adds a dictionnary to tile.paths with key "Sentinel" and with value another dictionnary where keys are ordered and formatted dates and values are the paths to the directories containing the different bands
    tile.paths["Sentinel"] = get_band_paths(tile.paths["Sentinel"]) #Replaces the paths to the directories for each date with a dictionnary where keys are the bands, and values are their paths
    
    #Adding directories for ouput. Directories are created and their paths added to the TileInfo object.
    tile.add_dirpath("VegetationIndexDir", tile.data_directory / "VegetationIndex")
    tile.add_dirpath("MaskDir", tile.data_directory / "Mask")
    if soil_detection:
        tile.add_path("state_soil", tile.data_directory / "DataSoil" / "state_soil.tif")
        tile.add_path("first_date_soil", tile.data_directory / "DataSoil" / "first_date_soil.tif")
        tile.add_path("count_soil", tile.data_directory / "DataSoil" / "count_soil.tif")
        
    #Computing cloudiness percentage for each date
    cloudiness = get_cloudiness(Path(input_directory) / "cloudiness", tile.paths["Sentinel"], sentinel_source) if lim_perc_cloud != -1 else dict(zip(tile.paths["Sentinel"], [-1]*len(tile.paths["Sentinel"]))) #Returns dictionnary with cloud percentage for each date, except if lim_perc_cloud is set as 1, in which case cloud percentage is -1 for every date so source mask is not used and every date is used 
    new_dates = np.array([date for date in tile.paths["Sentinel"] if cloudiness[date] <= lim_perc_cloud and date>=start_date and ((ignored_period is None or len(ignored_period) == 0) or (date[5:] > min(ignored_period) and date[5:] < max(ignored_period))) and (not(hasattr(tile, "dates")) or date > tile.dates[-1])]) #Creates array containing only the dates with cloudiness inferior to lim_perc_cloud parameter. Also filters out dates anterior to already used dates.
    tile.dates = np.concatenate((tile.dates, new_dates)) if hasattr(tile, "dates") else new_dates #Adds list of all used dates (already used + new dates) as attribute to TileInfo object
    tile.raster_meta = get_raster_metadata(list(tile.paths["Sentinel"].values())[-1][next(x for x in list(tile.paths["Sentinel"].values())[0] if x in ["B2","B3","B4","B8"])], #path of first 10m resolution band found
                                           extent_shape_path = extent_shape_path)  #Imports all raster metadata from one band. 

    if  len(new_dates) == 0:
        print("Computing masks and vegetation index : no new dates")
        tile.getdict_paths(path_vi = tile.paths["VegetationIndexDir"],
                            path_masks = tile.paths["MaskDir"])
        tile.save_info()
        return
    print("Computing masks and vegetation index : " + str(len(new_dates))+ " new dates")

    tile.used_bands, tile.vi_formula = get_bands_and_formula(vi, path_dict_vi = path_dict_vi, forced_bands = ["B2","B3","B4", "B8A","B11"] if soil_detection else get_bands_and_formula(formula = formula_mask)[0])

    date = new_dates[0]
    stack_bands = import_resampled_sen_stack(tile.paths["Sentinel"][date], tile.used_bands, interpolation_order = interpolation_order, extent = tile.raster_meta["extent"])

#Import or initialize data for the soil mask
    soil_data = None
    if soil_detection:
        if tile.paths["state_soil"].exists():
            soil_data = import_soil_data(tile.paths)
        else:
            shape = stack_bands[0].shape
            coords = stack_bands[0].coords
            soil_data = initialize_soil_data(shape,coords)


    f = process_batch_loop
    if multi_process:
        f = process_batch_multithread
    soil_data = f(tile, new_dates, soil_data, interpolation_order, sentinel_source, apply_source_mask, soil_detection, formula_mask, compress_raster, compress_vi, progress)
    if soil_detection:
        write_tif(soil_data["state"], tile.raster_meta["attrs"],tile.paths["state_soil"],nodata=0)
        write_tif(soil_data["first_date"], tile.raster_meta["attrs"],tile.paths["first_date_soil"],nodata=0)
        write_tif(soil_data["count"], tile.raster_meta["attrs"],tile.paths["count_soil"],nodata=0)

    tile.getdict_paths(path_vi = tile.paths["VegetationIndexDir"],
                        path_masks = tile.paths["MaskDir"])
    tile.save_info()
