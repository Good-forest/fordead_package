# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:52:12 2022

@author: rdutrieux, kose
"""
import time
import click
import geopandas as gp
from pathlib import Path
from validation_stac_module import get_reflectance_at_points, get_already_extracted
from stac_module import get_vectorBbox, getItemCollection
import os


# @click.command(name='extract_reflectance')
# @click.option("--obs_path", type = str,default = None, help = "Path to a vector file containing observation points, must have an ID column corresponding to name_column parameter, an 'area_name' column with the name of the Sentinel-2 tile from which to extract reflectance, and a 'espg' column containing the espg integer corresponding to the CRS of the Sentinel-2 tile.", show_default=True)
# @click.option("--sentinel_dir", type = str,default = None, help = "Path of the directory containing Sentinel-2 data.", show_default=True)
# @click.option("--export_path", type = str,default = None, help = "Path to write csv file with extracted reflectance", show_default=True)
# @click.option("--name_column", type = str,default = "id", help = "Name of the ID column", show_default=True)
# def cli_extract_reflectance(obs_path, sentinel_dir, export_path, name_column):
#     """
#     Extracts reflectance from Sentinel-2 data using a vector file containing points, exports the data to a csv file.
#     If new acquisitions are added to the Sentinel-2 directory, new data is extracted and added to the existing csv file.
    
#     \f
#     Parameters
#     ----------
#     obs_path : str
#         Path to a vector file containing observation points, must have an ID column corresponding to name_column parameter, an 'area_name' column with the name of the Sentinel-2 tile from which to extract reflectance, and a 'espg' column containing the espg integer corresponding to the CRS of the Sentinel-2 tile.
#     sentinel_dir : str
#         Path of the directory containing Sentinel-2 data.
#     export_path : str
#         Path to write csv file with extracted reflectance.
#     name_column : str
#         Name of the ID column. The default is "id".

#     """
    
#     start_time_debut = time.time()
#     extract_reflectance(obs_path, sentinel_dir, export_path, name_column)
#     print("Exporting reflectance : %s secondes ---" % (time.time() - start_time_debut))


def extract_reflectance(obs_path, item_collection, export_path, name_column = "id"):
    """
    Extracts reflectance from Sentinel-2 data using a vector file containing points, exports the data to a csv file.
    If new acquisitions are added to the Sentinel-2 directory, new data is extracted and added to the existing csv file.

    Parameters
    ----------
    obs_path : str
        Path to a vector file containing observation points, must have an ID column corresponding to name_column parameter, an 'area_name' column with the name of the Sentinel-2 tile from which to extract reflectance, and a 'espg' column containing the espg integer corresponding to the CRS of the Sentinel-2 tile.
    sentinel_dir : str
        Path of the directory containing Sentinel-2 data.
    export_path : str
        Path to write csv file with extracted reflectance.
    name_column : str, optional
        Name of the ID column. The default is "id".


    """
    
    obs = gp.read_file(obs_path)
    
    extracted_reflectance = get_already_extracted(export_path, obs, obs_path, name_column)

    reflectance = get_reflectance_at_points(obs, item_collection, extracted_reflectance, name_column)
    
    if reflectance is None:
        print("No new data to extract")
    else:
        print("Data extracted")
        reflectance.to_csv(export_path, mode='a', index=False, header=not(os.path.exists(export_path)))

if __name__ == '__main__':

    start_time = time.time()
    obs_path = "D:/PROJETS/PRJ_FORDEAD/TEST_STAC/data/export01.shp"
    export_path = "D:/PROJETS/PRJ_FORDEAD/TEST_STAC/data/export02.csv"

    obs_bbox = get_vectorBbox(obs_path)
    startDate = "2021-07-01"
    endDate = "2021-12-01"
    cloudPct = 20
    coll = getItemCollection(startDate, endDate, obs_bbox, cloudPct)

    extract_reflectance(obs_path, coll, export_path) #, name_column = "id")
    end_time = time.time()
    duration = end_time - start_time
    print(duration/60)