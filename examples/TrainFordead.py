# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:21:15 2020

@author: Raphaël Dutrieux
"""
#%% =============================================================================
#   LIBRAIRIES
# =============================================================================

import argparse
# from pathlib import Path
from fordead.ImportData import import_forest_mask, import_stackedmaskedVI, TileInfo
from fordead.ModelVegetationIndex import get_last_training_date, model_vi
from fordead.writing_data import write_tif
import time

def parse_command_line():
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_directory", dest = "data_directory",type = str,default = "C:/Users/admin/Documents/Deperissement/fordead_data/tests/OutputFordead/ZoneTest", help = "Dossier avec les indices de végétations et les masques")
    parser.add_argument("-s", "--threshold_anomaly", dest = "threshold_anomaly",type = float,default = 0.16, help = "Seuil minimum pour détection d'anomalies")
    parser.add_argument("-k", "--remove_outliers", dest = "remove_outliers", action="store_false",default = True, help = "Si activé, garde les outliers dans les deux premières années")
    parser.add_argument("-g", "--date_lim_training", dest = "date_lim_training",type = str,default = "2018-06-01", help = "Dernière date pouvant servir pour l'apprentissage")
    parser.add_argument("-l", "--min_last_date_training", dest = "min_last_date_training",type = str,default = "2018-01-01", help = "Première date de la détection")
    dictArgs={}
    for key, value in parser.parse_args()._get_kwargs():
    	dictArgs[key]=value
    
        
    return dictArgs


def trainfordead(
    data_directory,
    threshold_anomaly=0.16,
    remove_outliers=True,
    date_lim_training="2018-06-01",
    min_last_date_training="2018-01-01"
    ):

    start_time = time.time()
    
    tuile = TileInfo(data_directory)
    tuile.getdict_paths(path_vi = tuile.data_directory / "VegetationIndex",
                            path_masks = tuile.data_directory / "Mask",
                            path_forestmask = list((tuile.data_directory / "MaskForet").glob("*.tif"))[0])

    #Import du masque forêt
    forest_mask = import_forest_mask(tuile.paths["ForestMask"])
    
    # Import des index de végétations et des masques
    stack_vi, stack_masks = import_stackedmaskedVI(tuile,date_lim_learning=date_lim_training)
    
    # Déterminer la date à partir de laquelle il y a suffisamment de données valides (si elle existe)  
    last_training_date=get_last_training_date(stack_masks, forest_mask,
                                              min_last_date_training = min_last_date_training,
                                              nb_min_date = 10)
    
    #Fusion du masque forêt et des zones non utilisables par manque de données
    used_area_mask = forest_mask.where(last_training_date!=0,False)

    # Modéliser le CRSWIR tout en retirant outliers
    coeff_model = model_vi(stack_vi, stack_masks,used_area_mask, last_training_date,
                          threshold_anomaly=threshold_anomaly, remove_outliers=True)
    
    #Create missing directories and add paths to TileInfo object
    tuile.add_path("coeff_model", tuile.data_directory / "DataModel" / "coeff_model.tif")
    tuile.add_path("last_training_date", tuile.data_directory / "DataModel" / "Last_training_date.tif")
    tuile.add_path("used_area_mask", tuile.paths["ForestMask"].parent / "Used_area_mask.tif")

    #Ecrire rasters de l'index de la dernière date utilisée, les coefficients, la zone utilisable
    write_tif(last_training_date,stack_vi.attrs, tuile.paths["last_training_date"],nodata=0)
    write_tif(coeff_model,stack_vi.attrs, tuile.paths["coeff_model"])
    write_tif(used_area_mask,stack_vi.attrs, tuile.paths["used_area_mask"],nodata=0)
    #Save the TileInfo object
    tuile.save_info()

    print("Temps d execution : %s secondes ---" % (time.time() - start_time))
    
if __name__ == '__main__':
    dictArgs=parse_command_line()
    print(dictArgs)
    trainfordead(**dictArgs)
    

