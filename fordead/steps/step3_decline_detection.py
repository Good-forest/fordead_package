# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:25:23 2020

@author: Raphaël Dutrieux
"""


import argparse
import numpy as np
from fordead.ImportData import import_coeff_model, import_decline_data, initialize_decline_data, import_masked_vi, import_first_detection_date_index, TileInfo
from fordead.writing_data import write_tif
from fordead.decline_detection import detection_anomalies, prediction_vegetation_index, detection_decline
# import time


def parse_command_line():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_directory", dest = "data_directory",type = str, help = "Dossier avec les données")
    parser.add_argument("-s", "--threshold_anomaly", dest = "threshold_anomaly",type = float,default = 0.16, help = "Seuil minimum pour détection d'anomalies"),
    parser.add_argument("--vi", dest = "vi",type = str,default = None, help = "Chosen vegetation index, only useful if step1 was skipped")
    parser.add_argument("--path_dict_vi", dest = "path_dict_vi",type = str,default = None, help = "Path of text file to add vegetation index formula, only useful if step1 was skipped")

    dictArgs={}
    for key, value in parser.parse_args()._get_kwargs():
    	dictArgs[key]=value
    return dictArgs


def decline_detection(
    data_directory,
    threshold_anomaly=0.16,
    vi = None,
    path_dict_vi = None
    ):
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    tile.add_parameters({"threshold_anomaly" : threshold_anomaly})
    if tile.parameters["Overwrite"] : 
        tile.delete_dirs("AnomaliesDir","state_decline" ,"periodic_results_decline","result_files","timelapse","series") #Deleting previous detection results if they exist
        if hasattr(tile, "last_computed_anomaly"): delattr(tile, "last_computed_anomaly")
    if vi==None : vi = tile.parameters["vi"]
    if path_dict_vi==None : path_dict_vi = tile.parameters["path_dict_vi"] if "path_dict_vi" in tile.parameters else None
    
    tile.add_dirpath("AnomaliesDir", tile.data_directory / "DataAnomalies") #Choose anomalies directory
    tile.getdict_datepaths("Anomalies",tile.paths["AnomaliesDir"]) # Get paths and dates to previously calculated anomalies
    tile.search_new_dates() #Get list of all used dates
    
    tile.add_path("state_decline", tile.data_directory / "DataDecline" / "state_decline.tif")
    tile.add_path("first_date_decline", tile.data_directory / "DataDecline" / "first_date_decline.tif")
    tile.add_path("count_decline", tile.data_directory / "DataDecline" / "count_decline.tif")
    
    #Verify if there are new SENTINEL dates
    new_dates = tile.dates[tile.dates > tile.last_computed_anomaly] if hasattr(tile, "last_computed_anomaly") else tile.dates[tile.dates >= tile.parameters["min_last_date_training"]]
    if  len(new_dates) == 0:
        print("Decline detection : no new dates")
    else:
        print("Decline detection : " + str(len(new_dates))+ " new dates")
        
        #IMPORTING DATA
        first_detection_date_index = import_first_detection_date_index(tile.paths["first_detection_date_index"])
        coeff_model = import_coeff_model(tile.paths["coeff_model"])
        
        if tile.paths["state_decline"].exists():
            decline_data = import_decline_data(tile.paths)
        else:
            decline_data = initialize_decline_data(first_detection_date_index.shape,first_detection_date_index.coords)
        
        #DECLINE DETECTION
        for date_index, date in enumerate(tile.dates):
            if date in new_dates:
                masked_vi = import_masked_vi(tile.paths,date)
                masked_vi["mask"] = masked_vi["mask"] | (date_index < first_detection_date_index) #Masking pixels where date was used for training

                predicted_vi=prediction_vegetation_index(coeff_model,[date])
                
                anomalies = detection_anomalies(masked_vi["vegetation_index"], predicted_vi, threshold_anomaly, 
                                                vi = vi, path_dict_vi = path_dict_vi).squeeze("Time")
                                
                decline_data = detection_decline(decline_data, anomalies, masked_vi["mask"], date_index)
                               
                write_tif(anomalies, first_detection_date_index.attrs, tile.paths["AnomaliesDir"] / str("Anomalies_" + date + ".tif"),nodata=0)
        tile.last_computed_anomaly = new_dates[-1]
                
        #Writing decline data to rasters        
        write_tif(decline_data["state"], first_detection_date_index.attrs,tile.paths["state_decline"],nodata=0)
        write_tif(decline_data["first_date"], first_detection_date_index.attrs,tile.paths["first_date_decline"],nodata=0)
        write_tif(decline_data["count"], first_detection_date_index.attrs,tile.paths["count_decline"],nodata=0)
                
        # print("Détection du déperissement")
    tile.save_info()



if __name__ == '__main__':
    dictArgs=parse_command_line()
    # print(dictArgs)
    # start_time = time.time()
    decline_detection(**dictArgs)
    # print("Temps d execution : %s secondes ---" % (time.time() - start_time))
