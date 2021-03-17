# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:21:15 2020

@author: Raphaël Dutrieux
"""
#%% =============================================================================
#   LIBRAIRIES
# =============================================================================

import click
from fordead.ImportData import import_stackedmaskedVI, TileInfo
from fordead.ModelVegetationIndex import get_detection_dates, model_vi, model_vi_correction
from fordead.writing_data import write_tif


@click.command(name='train_model')
@click.option("-d", "--data_directory", type = str, help = "Path of the output directory")
@click.option("--nb_min_date", type = int,default = 10, help = "Minimum number of valid dates to compute a vegetation index model for the pixel", show_default=True)
@click.option("-s", "--min_last_date_training", type = str,default = "2018-01-01", help = "First date that can be used for detection", show_default=True)
@click.option("-e", "--max_last_date_training", type = str,default = "2018-06-01", help = "Last date that can be used for training", show_default=True)
@click.option("--correct_vi",  is_flag=True, help = "If True, corrects vi using large scale median vi", show_default=True)
@click.option("--path_vi", type = str,default = None, help = "Path of directory containing vegetation indices for each date. If None, the information has to be saved from a previous step", show_default=True)
@click.option("--path_masks", type = str,default = None, help = "Path of directory containing masks for each date.  If None, the information has to be saved from a previous step", show_default=True)
def cli_train_model(
    data_directory,
    nb_min_date = 10,
    min_last_date_training="2018-01-01",
    max_last_date_training="2018-06-01",
    correct_vi = False,
    path_vi=None,
    path_masks = None,
    ):
    """
    Uses first SENTINEL dates to train a periodic vegetation index model capable of predicting the vegetation index at any date.
    If there aren't nb_min_date at min_last_date_training, later dates between min_last_date_training and max_last_date_training can be used.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/02_train_model/
    
    \f

    Parameters
    ----------
    data_directory
    nb_min_date
    min_last_date_training
    max_last_date_training
    correct_vi
    path_vi
    path_masks

    Returns
    -------
    """
    train_model(data_directory,nb_min_date,min_last_date_training,max_last_date_training, correct_vi, path_vi,path_masks)


def train_model(
    data_directory,
    nb_min_date = 10,
    min_last_date_training="2018-01-01",
    max_last_date_training="2018-06-01",
    correct_vi = False,
    path_vi=None,
    path_masks = None,
    ):
    """
    Uses first SENTINEL dates to train a periodic vegetation index model capable of predicting the vegetation index at any date.
    If there aren't nb_min_date at min_last_date_training, later dates between min_last_date_training and max_last_date_training can be used.
    See details here : https://fordead.gitlab.io/fordead_package/docs/user_guides/02_train_model/
    \f

    Parameters
    ----------
    data_directory : str
        Path of the output directory
    nb_min_date : int
        Minimum number of valid dates to compute a vegetation index model for the pixel
    min_last_date_training : str
        First date that can be used for detection (format : 'YYYY-MM-DD')
    max_last_date_training : str
        Last date that can be used for training (format : 'YYYY-MM-DD')
    correct_vi : bool
        If True, corrects vi using large scale median vi
    path_vi : str
        Path of directory containing vegetation indices for each date. If None, the information has to be saved from a previous step
    path_masks : str
        Path of directory containing masks for each date.  If None, the information has to be saved from a previous step

    Returns
    -------

    """
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    
    if path_vi != None : tile.paths["VegetationIndexDir"] = path_vi
    if path_masks != None : tile.paths["MaskDir"] = path_masks

    
    tile.add_parameters({"nb_min_date" : nb_min_date, "min_last_date_training" : min_last_date_training, "max_last_date_training" : max_last_date_training, "correct_vi" : correct_vi})
    if tile.parameters["Overwrite"] : 
        tile.delete_dirs("coeff_model","AnomaliesDir","state_decline" ,"periodic_results_decline","result_files","timelapse","series", "validation") #Deleting previous training and detection results if they exist
        tile.delete_files("valid_area_mask")
        if hasattr(tile, "last_computed_anomaly"): delattr(tile, "last_computed_anomaly")
    #Create missing directories and add paths to TileInfo object
    tile.add_path("coeff_model", tile.data_directory / "DataModel" / "coeff_model.tif")
    tile.add_path("first_detection_date_index", tile.data_directory / "DataModel" / "first_detection_date_index.tif")
    tile.add_path("valid_area_mask", tile.data_directory / "ForestMask" / "valid_area_mask.tif")
    
    if tile.paths["coeff_model"].exists():
        print("Model already calculated")
    else:
        print("Computing model")
        tile.getdict_paths(path_vi = tile.paths["VegetationIndexDir"],
                            path_masks = tile.paths["MaskDir"])
        
        # Import des index de végétations et des masques
        stack_vi, stack_masks = import_stackedmaskedVI(tile, max_date=max_last_date_training, chunks = 1280)
   
        detection_dates, first_detection_date_index = get_detection_dates(stack_masks,
                                              min_last_date_training = min_last_date_training,
                                              nb_min_date = 10)
        
        
        #Fusion du masque forêt et des zones non utilisables par manque de données
        valid_area_mask = first_detection_date_index!=0
        
        if correct_vi:
            stack_vi, tile.large_scale_model, tile.correction_vi = model_vi_correction(stack_vi, tile.paths["ForestMask"])

        # Modéliser le CRSWIR
        stack_masks = stack_masks | detection_dates #Masking data not used in training
        coeff_model = model_vi(stack_vi, stack_masks)
        
        #Ecrire rasters de l'index de la dernière date utilisée, les coefficients, la zone utilisable
        write_tif(first_detection_date_index,tile.raster_meta["attrs"], tile.paths["first_detection_date_index"],nodata=0)
        write_tif(coeff_model,tile.raster_meta["attrs"], tile.paths["coeff_model"])
        write_tif(valid_area_mask,tile.raster_meta["attrs"], tile.paths["valid_area_mask"],nodata=0)
        #Save the TileInfo object
    tile.save_info()

    
if __name__ == '__main__':
    # print(dictArgs)
    # start_time = time.time()
    cli_train_model()
    # print("Temps d execution : %s secondes ---" % (time.time() - start_time))


