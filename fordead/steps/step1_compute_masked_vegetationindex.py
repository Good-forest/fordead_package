# -*- coding: utf-8 -*-

#%% =============================================================================
#   LIBRARIES
# =============================================================================

# import time
import click
from pathlib import Path
# import geopandas as gp
import numpy as np
from tqdm import tqdm
import warnings
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

@click.command(name='masked_vi')
@click.option("-i", "--input_directory", type = str, help = "Path of the directory with Sentinel dates")
@click.option("-o", "--data_directory", type = str, help = "Path of the output directory")
@click.option("-s", "--start_date", type = str,default = "2015-01-01", help = "First date of processing, dates before this date will be ignored.", show_default=True)
@click.option("-n", "--lim_perc_cloud", type = float,default = 0.4, help = "Maximum cloudiness at the tile scale, used to filter used SENTINEL dates. Set parameter as -1 to not filter based on cloudiness", show_default=True)
@click.option("--interpolation_order", type = int,default = 0, help ="interpolation order for bands at 20m resolution : 0 = nearest neighbour, 1 = linear, 2 = bilinéaire, 3 = cubique", show_default=True)
@click.option("--sentinel_source", type = str,default = "theia", help = "Source of data, can be 'theia' et 'scihub' et 'peps'", show_default=True)
@click.option("--apply_source_mask",  is_flag=True, help = "If True, applies the mask from SENTINEL-data supplier", show_default=True)
@click.option("--soil_detection",  is_flag=True, help = "If True, bare ground is detected and used as mask, but the process has not been tested on other data than THEIA data in France (see https://fordead.gitlab.io/fordead_package/docs/user_guides/english/01_compute_masked_vegetationindex/). If False, mask from formula_mask is applied.", show_default=True)
@click.option("--formula_mask", type = str,default = "(B2 >= 700)", help = "formula whose result would be binary, as described here https://fordead.gitlab.io/fordead_package/reference/fordead/masking_vi/#compute_vegetation_index. Is only used if soil_detection is False.", show_default=True)
@click.option("--vi", type = str,default = "CRSWIR", help = "Chosen vegetation index", show_default=True)
@click.option("--compress_vi",  is_flag=True, help = "Stores the vegetation index as low-resolution floating-point data as small integers in a netCDF file. Uses less disk space but can lead to very small difference in results as the vegetation is rounded to three decimal places", show_default=True)
@click.option("--ignored_period", multiple=True, type = str, default = None, help = "Period whose Sentinel dates to ignore (format 'MM-DD', ex : --ignored_period 11-01 --ignored_period 05-01", show_default=True)
@click.option("--extent_shape_path", type = str,default = None, help = "Path of shapefile used as extent of detection, if None, the whole tile is used", show_default=True)
@click.option("--path_dict_vi", type = str,default = None, help = "Path of text file to add vegetation index formula, if None, only built-in vegetation indices can be used (CRSWIR, NDVI)", show_default=True)
def cli_compute_masked_vegetationindex(**kwargs):
    """
    Computes masks and masked vegetation index for each SENTINEL date under a cloudiness threshold.
    The mask includes pixels ouside satellite swath, some shadows, and the mask from SENTINEL data provider if the option is chosen.
    Also, if soil detection is activated, the mask includes bare ground detection and cloud detection, but the process might only be adapted to THEIA data in France's coniferous forests.
    If it is not activated, then the user can choose a mask of his own using the formula_mask parameter.
    Results are written in the chosen directory.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/english/01_compute_masked_vegetationindex/
    \f

    """
    empty_to_none(kwargs, "ignored_period")
    compute_masked_vegetationindex(**kwargs)


def compute_masked_vegetationindex(
    input_directory,
    data_directory,
    start_date = "2015-01-01",
    lim_perc_cloud=0.4,
    interpolation_order = 0,
    sentinel_source = "theia",
    apply_source_mask = False,
    soil_detection = True,
    formula_mask = "(B2 >= 700)",
    vi = "CRSWIR",
    compress_vi = False,
    ignored_period = None,
    extent_shape_path=None,
    path_dict_vi = None,
    progress=True
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
        tile.delete_dirs("VegetationIndexDir", "MaskDir", "coeff_model", "AnomaliesDir",
                         "state_dieback", "state_soil", "periodic_results_dieback",
                         "result_files","timelapse","series", "nb_periods_stress")
        tile.delete_files("sufficient_coverage_mask","too_many_stress_periods_mask")
        tile.delete_attributes("last_computed_anomaly","dates","last_date_export")

    # All SENTINEL data in the input directory is detected, and paths are added to the TileInfo object. For example, after this operation tile.paths["Sentinel"]["YYYY-MM-DD"]["B2"] brings up the path to the B2 band file of the specified date?
    # check if upgrade is needed
    sentinel_bkp = None
    if "Sentinel" in tile.paths:
        sentinel_bkp = tile.paths["Sentinel"]
    tile.getdict_datepaths("Sentinel",Path(input_directory)) #adds a dictionnary to tile.paths with key "Sentinel" and with value another dictionnary where keys are ordered and formatted dates and values are the paths to the directories containing the different bands
    # warn if upgrade detected in sentinel source
    if sentinel_bkp is not None:
        for date in sentinel_bkp:
            if date in tile.paths["Sentinel"] and sentinel_bkp[date]["B8A"].parent.name != tile.paths["Sentinel"][date].name:
                warnings.warn(
                    "Input data has changed at " + str(date) + 
                    ". Remove previously computed results to update to new input source.")
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
    else:
        print("Computing masks and vegetation index : " + str(len(new_dates))+ " new dates")
        
        tile.raster_meta = get_raster_metadata(list(tile.paths["Sentinel"].values())[-1][next(x for x in list(tile.paths["Sentinel"].values())[1] if x in ["B2","B3","B4","B8"])], #path of first 10m resolution band found
                                               extent_shape_path = extent_shape_path)  #Imports all raster metadata from one band. 
        
        #Import or initialize data for the soil mask
        if soil_detection:
            if tile.paths["state_soil"].exists():
                soil_data = import_soil_data(tile.paths)
            else:
                soil_data = initialize_soil_data(tile.raster_meta["shape"],tile.raster_meta["coords"])

        tile.used_bands, tile.vi_formula = get_bands_and_formula(vi, path_dict_vi = path_dict_vi, forced_bands = ["B2","B3","B4", "B8A","B11"] if soil_detection else get_bands_and_formula(formula = formula_mask)[0]) #Selects only relevant bands depending on used vegetation index plus forced_bands used in masks
        
        for date_index, date in tqdm(enumerate(tile.dates), total=len(tile.dates), disable=not progress):
            if date in new_dates:
                
                # Resample and import SENTINEL data
                stack_bands = import_resampled_sen_stack(tile.paths["Sentinel"][date], tile.used_bands, interpolation_order = interpolation_order, extent = tile.raster_meta["extent"])
                
                # Compute vegetation index
                vegetation_index = compute_vegetation_index(stack_bands, formula = tile.vi_formula)
                invalid_values = vegetation_index.isnull() | np.isinf(vegetation_index)
                vegetation_index = vegetation_index.where(~invalid_values,0)

                # Compute mask
                if soil_detection:
                    mask = compute_masks(stack_bands, soil_data, date_index)
                else:
                    mask = compute_user_mask(stack_bands, formula_mask)
                mask = mask | invalid_values
                if apply_source_mask:
                    mask = mask | get_source_mask(tile.paths["Sentinel"][date], sentinel_source, extent = tile.raster_meta["extent"]) #Masking with source mask if option chosen
                
                #Writing vegetation index and mask
                # write_tif(vegetation_index, tile.raster_meta["attrs"],tile.paths["VegetationIndexDir"] / ("VegetationIndex_"+date+".nc"),nodata=0)
                write_tif(mask, tile.raster_meta["attrs"], tile.paths["MaskDir"] / ("Mask_"+date+".tif"),nodata=0)
                write_raster(vegetation_index, tile.paths["VegetationIndexDir"] / ("VegetationIndex_"+date+".nc"), compress_vi)
                # write_raster(mask, tile.paths["MaskDir"] / ("Mask_"+date+".nc"))

                del vegetation_index, mask
                # print('\r', date, " | ", len(tile.dates)-date_index-1, " remaining       ", sep='', end='', flush=True) if date_index != (len(tile.dates) -1) else print('\r', '                                              ', '\r', sep='', end='', flush=True)
            
        if soil_detection:
            #Writing soil data 
            write_tif(soil_data["state"], tile.raster_meta["attrs"],tile.paths["state_soil"],nodata=0)
            write_tif(soil_data["first_date"], tile.raster_meta["attrs"],tile.paths["first_date_soil"],nodata=0)
            write_tif(soil_data["count"], tile.raster_meta["attrs"],tile.paths["count_soil"],nodata=0)
            # write_raster(soil_data["state"],tile.paths["state_soil"])
            # write_raster(soil_data["first_date"],tile.paths["first_date_soil"])
            # write_raster(soil_data["count"],tile.paths["count_soil"])

    #Add paths to vi and mask to TileInfo object
    tile.getdict_paths(path_vi = tile.paths["VegetationIndexDir"],
                        path_masks = tile.paths["MaskDir"])
    
    #Saving TileInfo object
    tile.save_info()
    
if __name__ == '__main__':
    # start_time_debut = time.time()
    cli_compute_masked_vegetationindex()
    # print("Calcul des masques et du CRSWIR : %s secondes ---" % (time.time() - start_time_debut))
