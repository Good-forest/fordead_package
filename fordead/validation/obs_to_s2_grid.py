# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:52:12 2022

@author: rdutrieux
"""

import click
import geopandas as gp
import pandas as pd
from pathlib import Path
from fordead.validation_module import get_grid_points, process_points

@click.command(name='obs_to_s2_grid')
@click.option("--obs_path", type = str,default = None, help = "Path to a vector file containing observation points or polygons, must have an ID column corresponding to name_column parameter", show_default=True)
@click.option("--sentinel_dir", type = str,default = None, help = "Path of the directory containing Sentinel-2 data", show_default=True)
@click.option("--export_path", type = str,default = None, help = "Path used to write resulting vector file, with added 'epsg','area_name' and 'id_pixel' columns", show_default=True)
@click.option("--name_column", type = str,default = "id", help = "Name of the ID column", show_default=True)
@click.option("-t","--tile_selection", type = list, default = None, help = "A list of names of Sentinel-2 directories. (ex : -t T31UFQ -t T31UGQ). If None, all tiles are used.", show_default=True)
@click.option("--overwrite",  is_flag=True, help = "Overwrites file at obs_path", show_default=True)
def cli_obs_to_s2_grid(obs_path, sentinel_dir, export_path, name_column, tile_selection, overwrite):
    """
    Attributes intersecting Sentinel-2 tiles to observation points or polygons, adding their epsg and name. If polygons are used, they are converted to grid points located at the centroid of Sentinel-2 pixels.
    If points or polygons intersect several Sentinel-2 tiles, they are duplicated for each of them.
    If some intersect no Sentinel-2 tiles, they are removed and their IDs are printed.

    \f

    """
    
    obs_to_s2_grid(obs_path, sentinel_dir, export_path, name_column, tile_selection, overwrite)

def obs_to_s2_grid(obs_path, sentinel_dir, export_path, name_column = "id", tile_selection = None, overwrite = False):
    """
    Attributes intersecting Sentinel-2 tiles to observation points or polygons, adding their epsg and name. If polygons are used, they are converted to grid points located at the centroid of Sentinel-2 pixels.
    If points or polygons intersect several Sentinel-2 tiles, they are duplicated for each of them.
    If some intersect no Sentinel-2 tiles, they are removed and their IDs are printed.

    Parameters
    ----------
    obs_path : str
        Path to a vector file containing observation points or polygons, must have an ID column corresponding to name_column parameter.
    sentinel_dir : str
        Path of the directory containing Sentinel-2 data.
    export_path : str
        Path used to write resulting vector file, with added "epsg","area_name" and "id_pixel" columns.
    name_column : str, optional
        Name of the ID column. The default is "id".
    tile_selection : list
        A list of names of Sentinel-2 directories. If this parameter is used, extraction is limited to those directories.
    overwrite : bool
        If True, allows overwriting of file at obs_path

    """
    
    sentinel_dir = Path(sentinel_dir) ; export_path = Path(export_path)
    obs = gp.read_file(obs_path)

    if export_path.exists():
        print(str(export_path) + " already exists")
        if overwrite:
            print("It will be overwritten\n")
        else:
            raise Exception("Set 'overwrite' parameter as True to overwrite")
    
    geom_type = obs.geom_type.drop(columns = "geometry")
    points = obs[(geom_type == 'Point') | (geom_type == 'MultiPoint')]
    polygons = obs[(geom_type == 'Polygon') | (geom_type == 'MultiPolygon')]
    
    points_from_points = process_points(points, sentinel_dir, name_column, tile_selection) if len(points) != 0 else None
    points_from_poly = get_grid_points(polygons, sentinel_dir, name_column, tile_selection) if len(polygons) != 0 else None

    total_points = pd.concat([points_from_poly,points_from_points])
    total_points.to_file(export_path)

    
if __name__ == '__main__':

        obs_to_s2_grid(
            obs_path = "D:/fordead/fordead_data/vector/observations_tuto.shp",
            sentinel_dir = "D:/fordead/fordead_data/sentinel_data/validation_tutorial/sentinel_data", 
            export_path = "D:/fordead/fordead_data/output/pp_observations_tuto.shp",
            name_column = "id",
            overwrite = True)
