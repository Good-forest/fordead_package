# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:30:53 2020

@author: Raphael Dutrieux


Crée un timelapse à partir des résultats calculés


"""

#%% =============================================================================
#   LIBRAIRIES
# =============================================================================

# from glob import glob
import geopandas as gp
import click
from plotly.offline import plot
    
# %%=============================================================================
#  IMPORT LIBRAIRIES PERSO
# =============================================================================

from fordead.results_visualisation import CreateTimelapse
from fordead.ImportData import TileInfo

@click.group()
def timelapse():
    """Main entrypoint."""
    
@timelapse.command(name='timelapse')
@click.option("-o", "--data_directory", type = str, help = "Path of the directory containing results from the region of interest")
@click.option("--obs_terrain_path", type = str, help = "Path of the shapefile with ground observations")
@click.option("--shape_path", type = str, help = "Path of the shapefile of the area, or points, to convert to timelapse. Not used if timelapse made from coordinates")
@click.option("--name_column", type = str, default = "id", help = "Name of the column containing the name of the export. Not used if timelapse made from coordinates")
@click.option("--coordinates", type = tuple, help = "Tuple of coordinates in the crs of the Sentinel-2 tile. Format : (x,y). Not used if timelapse is made using a shapefile")
@click.option("--buffer", type = int, default = 100, help = "Buffer around polygons or points for the extent of the timelapse")
def cli_create_timelapse(data_directory, obs_terrain_path = None, shape_path = None, name_column = "id", coordinates = None, buffer = 100):
    """
    Create timelapse allowing navigation through Sentinel-2 dates with detection results superimposed.
    By specifying 'shape_path' and 'name_column' parameters, it can be used with a shapefile containing one or multiple polygons or points with a column containing a unique ID used to name the export. 
    By specifying 'coordinates' parameter, it can be used by specifying coordinates in the system of projection of the tile. 
    The timelapse is exported in the data_directory/Timelapses directory as an html file.
    See details https://fordead.gitlab.io/fordead_package/docs/user_guides/Results_visualization/
    \f
    Parameters
    ----------
    data_directory
    obs_terrain_path
    shape_path
    name_column
    coordinates
    buffer


    """
    create_timelapse(data_directory, obs_terrain_path, shape_path, name_column, coordinates, buffer)

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

def create_timelapse(data_directory, obs_terrain_path = None, shape_path = None, name_column = "id", coordinates = None, buffer = 100):
    """
    Create timelapse allowing navigation through Sentinel-2 dates with detection results superimposed.
    By specifying 'shape_path' and 'name_column' parameters, it can be used with a shapefile containing one or multiple polygons with a column containing a unique ID used to name the export. 
    By specifying 'coordinates' parameter, it can be used by specifying coordinates in the system of projection of the tile. 
    The timelapse is exported in the data_directory/Timelapses directory as an html file.
    See details https://fordead.gitlab.io/fordead_package/docs/user_guides/Results_visualization/

    Parameters
    ----------
    data_directory : str
        Path of the directory containing results from the region of interest.
    obs_terrain_path : str, optional
        Optionnal, Path of the shapefile with ground observations. The default is None.
    shape_path : str, optional
        Path of the shapefile of the area or points to convert to timelapse. Not used if timelapse made from coordinates. The default is None.
    name_column : str, optional
        Name of the column containing the name of the export. Not used if timelapse made from coordinates. The default is "id".
    coordinates : tuple, optional
        Tuple of coordinates in the crs of the Sentinel-2 tile. Format : (x,y). Not used if timelapse is made using shapefile. The default is None.
    buffer : int, optional
        Buffer around polygons or points for the extent of the timelapse. The default is 100.

    """
    
    
    tile = TileInfo(data_directory)
    tile = tile.import_info()
    # tile.add_parameters({"shape_path" : shape_path})
    # if tile.parameters["Overwrite"] : tile.delete_dirs("timelapse") #Deleting previous detection results if they exist
    tile.add_dirpath("timelapse", tile.data_directory / "Timelapses")
    tile.save_info()
    
    #Importing/creating ROI
    if coordinates is not None:
        print("Timelapse created from coordinates")
        ShapeInteret = gp.GeoDataFrame({"id" : [str(coordinates[0])+"_"+str(coordinates[1])]},geometry = gp.points_from_xy([coordinates[0]], [coordinates[1]]),crs =tile.raster_meta["attrs"]["crs"] )
        name_column = "id"
    elif shape_path is not None:
        print("Timelapse(s) created from " + shape_path)
        ShapeInteret=gp.read_file(shape_path)
        ShapeInteret=ShapeInteret.to_crs(crs = tile.raster_meta["attrs"]["crs"])
    else:
        raise Exception("No shape_path or coordinates")
        
    #Creating timelapse(s)
    for ShapeIndex in range(ShapeInteret.shape[0]):
        Shape=ShapeInteret.iloc[ShapeIndex:(ShapeIndex+1)]
        try:
            NameFile = str(Shape[name_column].iloc[0])
        except KeyError:
            raise Exception("No column "+name_column+" in " + shape_path)
        
        # if not((tile.paths["timelapse"] / (NameFile + ".html")).exists()):
        print("Creating timelapse | Id : " + NameFile)
        fig = CreateTimelapse(Shape.geometry.buffer(buffer),tile,DictCol, obs_terrain_path)
        plot(fig,filename=str(tile.paths["timelapse"] / (NameFile + ".html")),auto_open=False)


if __name__ == '__main__':
    cli_create_timelapse()

