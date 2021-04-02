# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:09:05 2021

@author: admin
"""


from examples.process_tile import process_tiles
import os

dep_path = "/mnt/fordead/Data/Vecteurs/Departements/departements-20140306-100m.shp"
tuiles = ["T31UFQ", "T31UFR", "T31UGP", "T31UGQ", "T31TGL"]
sentinel_directory = "/mnt/fordead/Data/SENTINEL/"
bdforet_dirpath = "/mnt/fordead/Data/Vecteurs/BDFORET"

vi = "NDVI"

for threshold_anomaly in [0.1,0.13,0.16]:
    main_directory = "/mnt/fordead/Out/"+vi+"_"+str(threshold_anomaly)
    os.mkdir(main_directory)
    process_tiles(main_directory = main_directory, sentinel_directory = sentinel_directory, tuiles = tuiles, forest_mask_source = "BDFORET", extent_shape_path = None,
                    dep_path = dep_path, bdforet_dirpath = bdforet_dirpath, list_forest_type =  ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"], path_oso = None, list_code_oso = None, #compute_forest_mask arguments
                    lim_perc_cloud = 0.5, vi = vi, sentinel_source = "THEIA", apply_source_mask = False, #compute_masked_vegetationindex arguments
                    min_last_date_training = "2018-01-01", date_lim_training = "2018-08-01", nb_min_date = 11,#Train_model arguments
                    threshold_anomaly = threshold_anomaly,
                    start_date_results = '2015-06-23', end_date_results = '2022-06-23', results_frequency = "M", multiple_files = False,
                    correct_vi = False, validation = True)

vi = "NDWI"

for threshold_anomaly in [0.12,0.14,0.16]:
    main_directory = "/mnt/fordead/Out/"+vi+"_"+str(threshold_anomaly)
    os.mkdir(main_directory)
    process_tiles(main_directory = main_directory, sentinel_directory = sentinel_directory, tuiles = tuiles, forest_mask_source = "BDFORET", extent_shape_path = None,
                    dep_path = dep_path, bdforet_dirpath = bdforet_dirpath, list_forest_type =  ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"], path_oso = None, list_code_oso = None, #compute_forest_mask arguments
                    lim_perc_cloud = 0.5, vi = vi, sentinel_source = "THEIA", apply_source_mask = False, #compute_masked_vegetationindex arguments
                    min_last_date_training = "2018-01-01", date_lim_training = "2018-08-01", nb_min_date = 11,#Train_model arguments
                    threshold_anomaly = threshold_anomaly,
                    start_date_results = '2015-06-23', end_date_results = '2022-06-23', results_frequency = "M", multiple_files = False,
                    correct_vi = False, validation = True)

threshold_anomaly = 0.15
main_directory = "/mnt/fordead/Out/"+vi+"_"+str(threshold_anomaly)+"_corrected"
os.mkdir(main_directory)
process_tiles(main_directory = main_directory, sentinel_directory = sentinel_directory, tuiles = tuiles, forest_mask_source = "BDFORET", extent_shape_path = None,
                dep_path = dep_path, bdforet_dirpath = bdforet_dirpath, list_forest_type =  ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"], path_oso = None, list_code_oso = None, #compute_forest_mask arguments
                lim_perc_cloud = 0.5, vi = vi, sentinel_source = "THEIA", apply_source_mask = False, #compute_masked_vegetationindex arguments
                min_last_date_training = "2018-01-01", date_lim_training = "2018-08-01", nb_min_date = 11,#Train_model arguments
                threshold_anomaly = threshold_anomaly,
                start_date_results = '2015-06-23', end_date_results = '2022-06-23', results_frequency = "M", multiple_files = False,
                correct_vi = True, validation = True)

threshold_anomaly = 0.15
vi = "CRSWIR"
main_directory = "/mnt/fordead/Out/"+vi+"_"+str(threshold_anomaly)+"_corrected"
os.mkdir(main_directory)
process_tiles(main_directory = main_directory, sentinel_directory = sentinel_directory, tuiles = tuiles, forest_mask_source = "BDFORET", extent_shape_path = None,
                dep_path = dep_path, bdforet_dirpath = bdforet_dirpath, list_forest_type =  ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"], path_oso = None, list_code_oso = None, #compute_forest_mask arguments
                lim_perc_cloud = 0.5, vi = vi, sentinel_source = "THEIA", apply_source_mask = False, #compute_masked_vegetationindex arguments
                min_last_date_training = "2018-01-01", date_lim_training = "2018-08-01", nb_min_date = 11,#Train_model arguments
                threshold_anomaly = threshold_anomaly,
                start_date_results = '2015-06-23', end_date_results = '2022-06-23', results_frequency = "M", multiple_files = False,
                correct_vi = True, validation = True)

threshold_anomaly = 0.16
vi = "CRSWIR"
main_directory = "/mnt/fordead/Out/"+vi+"_"+str(threshold_anomaly)
os.mkdir(main_directory)
process_tiles(main_directory = main_directory, sentinel_directory = sentinel_directory, tuiles = tuiles, forest_mask_source = "BDFORET", extent_shape_path = None,
                dep_path = dep_path, bdforet_dirpath = bdforet_dirpath, list_forest_type =  ["FF2-00-00", "FF2-90-90", "FF2-91-91", "FF2G61-61"], path_oso = None, list_code_oso = None, #compute_forest_mask arguments
                lim_perc_cloud = 0.5, vi = vi, sentinel_source = "THEIA", apply_source_mask = False, #compute_masked_vegetationindex arguments
                min_last_date_training = "2018-01-01", date_lim_training = "2018-08-01", nb_min_date = 11,#Train_model arguments
                threshold_anomaly = threshold_anomaly,
                start_date_results = '2015-06-23', end_date_results = '2022-06-23', results_frequency = "M", multiple_files = False,
                correct_vi = False, validation = True)