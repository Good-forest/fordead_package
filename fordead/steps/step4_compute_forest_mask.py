# -*- coding: utf-8 -*-

import click
import logging
from fordead.cli.utils import empty_to_none

logger = logging.getLogger(__name__)
from fordead.import_data import TileInfo, import_binary_raster, get_raster_metadata, clip_xarray
from fordead.masking_vi import rasterize_bdforet, clip_oso, raster_full, rasterize_vector
from fordead.writing_data import write_tif
from pathlib import Path

@click.command(name='forest_mask')
@click.option("-o", "--data_directory",  type=str, help="Path of the output directory", show_default=True)
@click.option("-f", "--forest_mask_source",  type=str, default=None,
                    help="Source of the forest mask, can be 'vector' to use a vector file at vector_path, or the path to a binary raster of 10m resolution with the value 1 on the pixels of interest, 'BDFORET' to use the BD Foret of the IGN, 'OSO' to use the CESBIO's land use map, or None to not use a forest mask and to extend the area of interest to all pixels", show_default=True)
@click.option("--dep_path",  type=str,
                    help="Path to shapefile containg departements with code insee. Optionnal, only used if forest_mask_source equals 'BDFORET'", show_default=True)
@click.option("--bdforet_dirpath",  type=str,
                    help="Path to directory containing BD FORET. Optionnal, only used if forest_mask_source equals 'BDFORET'", show_default=True)
@click.option("--list_forest_type", multiple=True, type=str,
                    default=["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"],
                    help="List of forest types to be kept in the forest mask, corresponds to the CODE_TFV of the BD FORET. Optionnal, only used if forest_mask_source equals 'BDFORET'", show_default=True)
@click.option("--path_oso",  type=str,
                    help="Path to soil occupation raster, only used if forest_mask_source = 'OSO' ", show_default=True)
@click.option("--list_code_oso", multiple=True,  type=int, default=[17],
                    help="List of values used to filter the soil occupation raster. Only used if forest_mask_source = 'OSO'", show_default=True)
@click.option("--vector_path",  type=str,
                    help="path of shapefile whose polygons will be rasterized as a binary raster with resolution, extent and crs of the raster at path_example_raster. Only used if forest_mask_source = 'vector'", show_default=True)
@click.option("--path_example_raster",  type=str, default=None,
                    help="Path to raster from which to copy the extent, resolution, CRS...", show_default=True)
def cli_compute_forest_mask(**kwargs):
    """
    Compute forest mask from IGN's BDFORET or CESBIO's OSO map
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/english/04_compute_forest_mask/
    \f

    """
    empty_to_none(kwargs, "list_forest_type")
    empty_to_none(kwargs, "list_code_oso")
    compute_forest_mask(**kwargs)


def compute_forest_mask(data_directory,
                        forest_mask_source = None,

                        list_forest_type = ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"],
                        dep_path = None,
                        bdforet_dirpath = None,
                        
                        path_oso = None,
                        list_code_oso = [17],
                        vector_path = None,
                        path_example_raster = None
                        ):
    """
    Compute forest mask
    \f
    Parameters
    ----------
    data_directory : str
        Path of the output directory
    forest_mask_source : str
        Source of the forest mask, can be 'vector' to use a vector file at vector_path, or the path to a binary raster of 10m resolution with the value 1 on the pixels of interest, 'BDFORET' to use the BD Foret of the IGN, 'OSO' to use the CESBIO's land use map, or None to not use a forest mask and to extend the area of interest to all pixels
    list_forest_type : list
        List of forest types to be kept in the forest mask, corresponds to the CODE_TFV of the BD FORET. Optionnal, only used if forest_mask_source equals 'BDFORET'
    dep_path : str
        Path to shapefile containg departements with code insee. Optionnal, only used if forest_mask_source equals 'BDFORET'
    bdforet_dirpath : str
        Path to directory containing BD FORET. Optionnal, only used if forest_mask_source equals 'BDFORET'
    path_oso : str
        Path to soil occupation raster, only used if forest_mask_source = 'OSO'
    list_code_oso : list
        List of values used to filter the soil occupation raster. Only used if forest_mask_source = 'OSO'
    vector_path : str
        path of shapefile whose polygons will be rasterized as a binary raster with resolution, extent and crs of the raster at path_example_raster. Only used if forest_mask_source = "vector"
    path_example_raster : str
        Path to raster from which to copy the extent, CRS...

    Returns
    -------

    """
    
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    tile.add_parameters({"forest_mask_source" : forest_mask_source, "list_forest_type" : list_forest_type, "list_code_oso" : list_code_oso, "vector_path" : vector_path})
    if tile.parameters["Overwrite"] : 
        tile.delete_files("forest_mask" ,"periodic_results_dieback","result_files","timelapse")
        tile.delete_attributes("last_date_export")
        #Si correction de l'indice de végétation, le calcul du masque forêt se fait en step2 et d'autres résultats doivent être supprimés
        if hasattr(tile, "correct_vi") and tile.parameters["correct_vi"] : 
            tile.delete_dirs("coeff_model","AnomaliesDir","state_dieback" ,"periodic_results_dieback","result_files","timelapse","series", "validation", "nb_periods_stress") #Deleting previous training and detection results if they exist
            tile.delete_files("sufficient_coverage_mask","too_many_stress_periods_mask")
            tile.delete_attributes("last_computed_anomaly")


    tile.add_path("forest_mask", tile.data_directory / "TimelessMasks" / "Forest_Mask.tif")

    
    if tile.paths["forest_mask"].exists():
        logger.info("Forest mask already calculated")
        tile.save_info()
    else:
        if path_example_raster is None : path_example_raster = tile.paths["Masks"][tile.dates[-1]]
        
        if forest_mask_source is None:
            logger.info("No mask used, computing forest mask with every pixel marked as True")
            forest_mask = raster_full(path_example_raster, fill_value = 1, dtype = bool)
            
        elif Path(forest_mask_source).is_file():
            logger.info("Importing " + forest_mask_source)
            forest_mask = clip_xarray(array = import_binary_raster(forest_mask_source), 
                                      extent = get_raster_metadata(path_example_raster)["extent"])
            
        elif forest_mask_source=="BDFORET":
            logger.info("Computing forest mask from BDFORET")
            forest_mask = rasterize_bdforet(path_example_raster, dep_path, bdforet_dirpath, list_forest_type = list_forest_type)
            
        elif  forest_mask_source=="OSO":
            logger.info("Computing forest mask from OSO")
            forest_mask = clip_oso(path_oso, path_example_raster, list_code_oso)
            
        elif forest_mask_source == "vector":
            logger.info("Computing mask from vector")
            forest_mask = rasterize_vector(vector_path, path_example_raster)
            
        else:
            logger.error("Unrecognized forest_mask_source")
        
        
        write_tif(forest_mask, forest_mask.attrs, nodata = 0, path = tile.paths["forest_mask"])
        tile.save_info()
        
if __name__ == '__main__':
    # print(dictArgs)
    # start_time_debut = time.time()
    cli_compute_forest_mask()
    # print("Computing forest mask : %s secondes ---" % (time.time() - start_time_debut))
