# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:30:53 2020

@author: admin


Crée un timelapse à partir des résultats calculés


"""

#%% =============================================================================
#   LIBRAIRIES
# =============================================================================

# from glob import glob
import geopandas as gp
import argparse
# import time
from plotly.offline import plot
    
# %%=============================================================================
#  IMPORT LIBRAIRIES PERSO
# =============================================================================

from fordead.timelapse import CreateTimelapse
from fordead.ImportData import TileInfo

# =============================================================================
#  CHOIX PARAMETRES
# =============================================================================

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_directory", dest = "data_directory",type = str,default = "D:/Documents/Deperissement/Output/ZoneTest2", help = "Directory with the computed results of decline detection")
    parser.add_argument("-s", "--shape_path", dest = "shape_path",type = str,default = "C:/Users/admin/Documents/Deperissement/fordead_data/Vecteurs/ZoneTest2.shp", help = "Path of the shapefile of the area to convert to timelapse")
    parser.add_argument("--obs_terrain_path", dest = "obs_terrain_path",type = str,default = "C:/Users/admin/Documents/Deperissement/fordead_data/Vecteurs/ObservationsTerrain/Scolytes.shp", help = "Path of the shapefile with ground observations")
    dictArgs={}
    for key, value in parser.parse_args()._get_kwargs():
    	dictArgs[key]=value
    return dictArgs

#%% =============================================================================
#   MAIN CODE
# =============================================================================
       
DictCol={'C' : "white",
         'V' : "lawngreen",
         "R" : "red",
         'S' : "black",
         'I' : "darkgreen",
         'G' : "darkgray",
         'X' : "indianred"}

#DictColAtteint={1 : "yellow",
#         2 : "black",
#         3 : "blue"}

def create_timelapse(data_directory,shape_path, obs_terrain_path, Overwrite):
    
    tile = TileInfo(data_directory)
    tile = tile.import_info()

    tile.add_dirpath("timelapse", tile.data_directory / "Timelapses")
    
    ShapeInteret=gp.read_file(shape_path)

    for ShapeIndex in range(ShapeInteret.shape[0]):
        Shape=ShapeInteret.iloc[ShapeIndex:(ShapeIndex+1)]
        if 'Id' in Shape.columns:
            NameFile=str(Shape["Id"].iloc[0])
        else:
            NameFile=str(ShapeIndex)
        print(NameFile)
        
        if Overwrite or not((tile.paths["timelapse"] / (NameFile + ".html")).exists()):
            fig = CreateTimelapse(Shape,tile,DictCol, obs_terrain_path)
            plot(fig,filename=str(tile.paths["timelapse"] / (NameFile + ".html")),auto_open=True)


if __name__ == '__main__':
    dictArgs=parse_command_line()
    # print(dictArgs)
    create_timelapse(**dictArgs)

