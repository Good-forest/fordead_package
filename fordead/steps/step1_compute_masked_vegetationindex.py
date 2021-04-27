# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:06:51 2020

@author: Raphaël Dutrieux

"""

#%% =============================================================================
#   LIBRAIRIES
# =============================================================================

# import time
import click
from pathlib import Path
# import geopandas as gp
import numpy as np
#%% ===========================================================================
#   IMPORT LIBRAIRIES PERSO
# =============================================================================

from fordead.ImportData import TileInfo, get_band_paths, get_cloudiness, import_resampled_sen_stack, import_soil_data, initialize_soil_data, get_raster_metadata
from fordead.masking_vi import compute_masks, compute_vegetation_index, get_bands_and_formula, get_source_mask
from fordead.writing_data import write_tif

#%% =============================================================================
#   FONCTIONS
# =============================================================================

@click.command(name='masked_vi')
@click.option("-i", "--input_directory", type = str, help = "Path of the directory with Sentinel dates")
@click.option("-o", "--data_directory", type = str, help = "Path of the output directory")
@click.option("-n", "--lim_perc_cloud", type = float,default = 0.4, help = "Maximum cloudiness at the tile scale, used to filter used SENTINEL dates. Set parameter as -1 to not filter based on cloudiness", show_default=True)
@click.option("--interpolation_order", type = int,default = 0, help ="interpolation order for bands at 20m resolution : 0 = nearest neighbour, 1 = linear, 2 = bilinéaire, 3 = cubique", show_default=True)
@click.option("--sentinel_source", type = str,default = "THEIA", help = "Source of data, can be 'THEIA' et 'Scihub' et 'PEPS'", show_default=True)
@click.option("--apply_source_mask",  is_flag=True, help = "If True, applies the mask from SENTINEL-data supplier", show_default=True)
@click.option("--vi", type = str,default = "CRSWIR", help = "Chosen vegetation index", show_default=True)
@click.option("--extent_shape_path", type = str,default = None, help = "Path of shapefile used as extent of detection, if None, the whole tile is used", show_default=True)
@click.option("--path_dict_vi", type = str,default = None, help = "Path of text file to add vegetation index formula, if None, only built-in vegetation indices can be used (CRSWIR, NDVI)", show_default=True)
def cli_compute_masked_vegetationindex(
    input_directory,
    data_directory,
    lim_perc_cloud=0.4,
    interpolation_order = 0,
    sentinel_source = "THEIA",
    apply_source_mask = False,
    vi = "CRSWIR",
    extent_shape_path=None,
    path_dict_vi = None
    ):
    """
    Computes masks and masked vegetation index for each SENTINEL date under a cloudiness threshold.
    Masks include shadows, clouds, soil, pixels ouside satellite swath, and the mask from SENTINEL data provider if the option is chosen.
    Results are written in the chosen directory.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/01_compute_masked_vegetationindex/
    \f
    Parameters
    ----------
    input_directory
    data_directory
    lim_perc_cloud
    interpolation_order
    sentinel_source
    apply_source_mask
    vi
    extent_shape_path
    path_dict_vi

    Returns
    -------

    """
    compute_masked_vegetationindex(input_directory,data_directory, lim_perc_cloud, interpolation_order, sentinel_source, apply_source_mask, vi, extent_shape_path, path_dict_vi)


def compute_masked_vegetationindex(
    input_directory,
    data_directory,
    lim_perc_cloud=0.4,
    interpolation_order = 0,
    sentinel_source = "THEIA",
    apply_source_mask = False,
    vi = "CRSWIR",
    extent_shape_path=None,
    path_dict_vi = None
    ):
    """
    Computes masks and masked vegetation index for each SENTINEL date under a cloudiness threshold.
    Masks include shadows, clouds, soil, pixels ouside satellite swath, and the mask from SENTINEL data provider if the option is chosen.
    Results are written in the chosen directory.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/01_compute_masked_vegetationindex/
    
    Parameters
    ----------
    input_directory : str
        Path of the directory with Sentinel dates
    data_directory : str
        Path of the output directory
    lim_perc_cloud : float
        Maximum cloudiness at the tile scale, used to filter used SENTINEL dates. Set parameter as -1 to not filter based on cloudiness
    interpolation_order : int
        interpolation order for bands at 20m resolution : 0 = nearest neighbour, 1 = linear, 2 = bilinéaire, 3 = cubique
    sentinel_source : str
        Source of data, can be 'THEIA' et 'Scihub' et 'PEPS'
    apply_source_mask : bool
        If True, applies the mask from SENTINEL-data supplier
    vi : str
        Chosen vegetation index
    extent_shape_path : str
        Path of shapefile used as extent of detection, if None, the whole tile is used
    path_dict_vi : str
        Path of text file to add vegetation index formula, if None, only built-in vegetation indices can be used (CRSWIR, NDVI)

    """
    if extent_shape_path is not None: data_directory = Path(data_directory).parent / Path(extent_shape_path).stem
    
    # Creation of TileInfo object. If it already exists in the specified directory, it is imported. 
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    
    # Parameters used are added to the TileInfo Object
    tile.add_parameters({"lim_perc_cloud" : lim_perc_cloud, "interpolation_order" : interpolation_order, "sentinel_source" : sentinel_source, "apply_source_mask" : apply_source_mask, "vi" : vi, "extent_shape_path" : extent_shape_path, "path_dict_vi" : path_dict_vi})
  
    # If parameters added differ from previously used parameters, all previous computation results are deleted
    if tile.parameters["Overwrite"] : 
        tile.delete_dirs("VegetationIndexDir", "MaskDir","coeff_model", "AnomaliesDir","state_decline", "state_soil" ,"periodic_results_decline","result_files","timelapse","series")
        tile.delete_files("valid_area_mask")
        if hasattr(tile, "last_computed_anomaly"): delattr(tile, "last_computed_anomaly")
        if hasattr(tile, "last_date_confidence_index"): delattr(tile, "last_date_confidence_index")
        if hasattr(tile, "dates"): delattr(tile, "dates")
        
    # All SENTINEL data in the input directory is detected, and paths are added to the TileInfo object. For example, after this operation tile.paths["Sentinel"]["YYYY-MM-DD"]["B2"] brings up the path to the B2 band file of the specified date?
    tile.getdict_datepaths("Sentinel",Path(input_directory)) #adds a dictionnary to tile.paths with key "Sentinel" and with value another dictionnary where keys are ordered and formatted dates and values are the paths to the directories containing the different bands
    tile.paths["Sentinel"] = get_band_paths(tile.paths["Sentinel"]) #Replaces the paths to the directories for each date with a dictionnary where keys are the bands, and values are their paths
    
    #Adding directories for ouput. Directories are created and their paths added to the TileInfo object.
    tile.add_dirpath("VegetationIndexDir", tile.data_directory / "VegetationIndex")
    tile.add_dirpath("MaskDir", tile.data_directory / "Mask")
    tile.add_path("state_soil", tile.data_directory / "DataSoil" / "state_soil.tif")
    tile.add_path("first_date_soil", tile.data_directory / "DataSoil" / "first_date_soil.tif")
    tile.add_path("count_soil", tile.data_directory / "DataSoil" / "count_soil.tif")
        
    #Computing cloudiness percentage for each date
    cloudiness = get_cloudiness(Path(input_directory) / "cloudiness", tile.paths["Sentinel"], sentinel_source) if lim_perc_cloud != -1 else dict(zip(tile.paths["Sentinel"], [-1]*len(tile.paths["Sentinel"]))) #Returns dictionnary with cloud percentage for each date, except if lim_perc_cloud is set as 1, in which case cloud percentage is -1 for every date so source mask is not used and every date is used 
    new_dates = np.array([date for date in tile.paths["Sentinel"] if cloudiness[date] <= lim_perc_cloud and (not(hasattr(tile, "dates")) or date > tile.dates[-1])]) #Creates array containing only the dates with cloudiness inferior to lim_perc_cloud parameter. Also filters out dates anterior to already used dates.
    tile.dates = np.concatenate((tile.dates, new_dates)) if hasattr(tile, "dates") else new_dates #Adds list of all used dates (already used + new dates) as attribute to TileInfo object
    
    if  len(new_dates) == 0:
        print("Computing masks and vegetation index : no new dates")
    else:
        print("Computing masks and vegetation index : " + str(len(new_dates))+ " new dates")
        
        tile.raster_meta = get_raster_metadata(list(tile.paths["Sentinel"].values())[0]["B2"], extent_shape_path = extent_shape_path)  #Imports all raster metadata from one band. 

        #Import or initialize data for the soil mask
        if tile.paths["state_soil"].exists():
            soil_data = import_soil_data(tile.paths)
        else:
            soil_data = initialize_soil_data(tile.raster_meta["shape"],tile.raster_meta["coords"])

        tile.used_bands, tile.vi_formula = get_bands_and_formula(vi, path_dict_vi, forced_bands = ["B2","B3","B4","B11","B8A"]) #Selects only relevant bands depending on used vegetation index plus forced_bands used in masks
    
        for date_index, date in enumerate(tile.dates):
            if date in new_dates:
                
                
                # Resample and import SENTINEL data
                stack_bands = import_resampled_sen_stack(tile.paths["Sentinel"][date], tile.used_bands, interpolation_order = interpolation_order, extent = tile.raster_meta["extent"])
                # Compute masks
                mask = compute_masks(stack_bands, soil_data, date_index)
                if apply_source_mask:
                    mask = mask | get_source_mask(tile.paths["Sentinel"][date], sentinel_source, extent = tile.raster_meta["extent"]) #Masking with source mask if option chosen
                    
                # Compute vegetation index
                vegetation_index = compute_vegetation_index(stack_bands, formula = tile.vi_formula)

                # Masking invalid values (division by zero)
                invalid_values = vegetation_index.isnull() | np.isinf(vegetation_index)
                vegetation_index = vegetation_index.where(~invalid_values,0)
                mask = mask | invalid_values
                
                #Writing vegetation index and mask
                write_tif(vegetation_index, tile.raster_meta["attrs"],tile.paths["VegetationIndexDir"] / ("VegetationIndex_"+date+".tif"),nodata=0)
                write_tif(mask, tile.raster_meta["attrs"], tile.paths["MaskDir"] / ("Mask_"+date+".tif"),nodata=0)
                del vegetation_index, mask
                print('\r', date, " | ", len(tile.dates)-date_index-1, " remaining       ", sep='', end='', flush=True) if date_index != (len(tile.dates) -1) else print('\r', '                                              ', '\r', sep='', end='', flush=True)
            
        #Writing soil data 
        write_tif(soil_data["state"], tile.raster_meta["attrs"],tile.paths["state_soil"],nodata=0)
        write_tif(soil_data["first_date"], tile.raster_meta["attrs"],tile.paths["first_date_soil"],nodata=0)
        write_tif(soil_data["count"], tile.raster_meta["attrs"],tile.paths["count_soil"],nodata=0)

    #Add paths to vi and mask to TileInfo object
    tile.getdict_paths(path_vi = tile.paths["VegetationIndexDir"],
                        path_masks = tile.paths["MaskDir"])
    #Saving TileInfo object
    tile.save_info()
    
if __name__ == '__main__':
    # start_time_debut = time.time()
    cli_compute_masked_vegetationindex()
    # print("Calcul des masques et du CRSWIR : %s secondes ---" % (time.time() - start_time_debut))


