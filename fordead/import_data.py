# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:42:31 2020

@author: Raphael Dutrieux
"""

import numpy as np
import xarray as xr
import re
# import datetime
from pathlib import Path
import pickle
import shutil
from scipy import ndimage
import geopandas as gp

def get_band_paths(dict_sen_paths):
    """
    Retrieves paths to each SENTINEL band for each date from the paths of the directories containing these bands for each date.

    Parameters
    ----------
    dict_sen_paths : dict
        dictionnary where keys are dates and values are the paths of the directory containing a file for each SENTINEL band

    Returns
    -------
    DictSentinelPaths : dict
        dictionnary with the same keys as dict_sen_paths, but where the paths to directories are replaced with another dictionnary where keys are the name of the bands, and values are their paths.

    """
    
    DictSentinelPaths={}
    for date in dict_sen_paths:
        AllPaths = dict_sen_paths[date].glob("**/*.tif")
        DictSentinelPaths[date]={}
        for path in AllPaths:
            path=str(path)
            if "B2" in path or "B02" in path:
                DictSentinelPaths[date]["B2"]=Path(path)
            if "B3" in path or "B03" in path:
                DictSentinelPaths[date]["B3"]=Path(path)
            if "B4" in path or "B04" in path:
                DictSentinelPaths[date]["B4"]=Path(path)
            if "B5" in path or "B05" in path:
                DictSentinelPaths[date]["B5"]=Path(path)
            if "B6" in path or "B06" in path:
                DictSentinelPaths[date]["B6"]=Path(path)
            if "B7" in path or "B07" in path:
                DictSentinelPaths[date]["B7"]=Path(path)
            if ("B8" in path or "B08" in path) and not("B8A" in path):
                DictSentinelPaths[date]["B8"]=Path(path)
            if "B8A" in path:
                DictSentinelPaths[date]["B8A"]=Path(path)
            if "B11" in path:
                DictSentinelPaths[date]["B11"]=Path(path)
            if "B12" in path:
                DictSentinelPaths[date]["B12"]=Path(path)
                
            if "_CLM_" in path or "SCL" in path:
                DictSentinelPaths[date]["Mask"]=Path(path)
   
    return DictSentinelPaths


def retrieve_date_from_string(string):
    """
    From a string containing a date in the format YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, DD-MM-YYYY, DD_MM_YYYY or DDMMYYYY, retrieves the date in the format YYYY-MM-DD.
    Works only for 20th and 21st centuries (years beginning with 19 or 20)

    Parameters
    ----------
    string : str
        String containing a date

    Returns
    -------
    formatted_date : str
        Date in the format YYYY-MM-DD

    """
    matchDMY = re.search(r'[0-3]\d-[0-1]\d-(19|20)\d{2}|[0-3]\d_[0-1]\d_(19|20)\d{2}|[0-3]\d[0-1]\d(19|20)\d{2}', string)
    matchYMD = re.search(r'(19|20)\d{2}-[0-1]\d-[0-3]\d|(19|20)\d{2}_[0-1]\d_[0-3]\d|(19|20)\d{2}[0-1]\d[0-3]\d', string)
    if matchDMY!=None:
        raw_date=matchDMY.group()
        if len(raw_date)==10:
            formatted_date=raw_date[-4:]+"-"+raw_date[3:5]+"-"+raw_date[:2]
        elif len(raw_date)==8:
            formatted_date=raw_date[-4:]+"-"+raw_date[2:4]+"-"+raw_date[:2]
    elif matchYMD!=None:
        raw_date=matchYMD.group()
        if len(raw_date)==10:
            formatted_date=raw_date[:4]+"-"+raw_date[5:7]+"-"+raw_date[-2:]
        elif len(raw_date)==8:
            formatted_date=raw_date[:4]+"-"+raw_date[4:6]+"-"+raw_date[-2:]
    else:
        return None
            
    return formatted_date

class TileInfo:
    def __init__(self, data_directory):
        """
        Initialize TileInfo object. This object is meant to store all relevant information (paths to input and output data, parameters used, used SENTINEL dates)
        
        Parameters
        ----------
        data_directory : str
            Path of a directory. This directory will be created if it doesn't exist, it is meant to be where the TileInfo object will be saved, though it doesn't have to be. 

        Returns
        -------
        None.

        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)   
        self.paths={}
        
        # print(globals())


    def import_info(self, path= None):
        """
        Imports TileInfo object in the data_directory, or the one at path if the parameter is given
        If no TileInfo object exists, the object remains unchanged
        
        Parameters
        ----------
        path : str, optional
            Path to a TileInfo object to be imported. The default is None.

        Returns
        -------
        TileInfo object
            Imported TileInfo object if one exists already, or current TileInfo object if not.
        """
        
        if path is None:
            path = self.data_directory / "TileInfo"
        if path.exists():
            with open(path, 'rb') as f:
                tuile = pickle.load(f)
                return tuile
        else:
            return self

    def save_info(self, path= None):
        """
        Saves the TileInfo object in its data_directory by default or in specified location
        """
        
        if path==None:
            path=self.data_directory / "TileInfo"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def print_info(self):
        """
        Prints parameters, dates used, and last computed date for anomalies.

        """
        
        if hasattr(self, "parameters"):
            print(" PARAMETRES\n")
            for parameter in self.parameters:
                print(parameter + " : " + str(self.parameters[parameter]))
            
        if hasattr(self, "dates"):
            print("\n " + str(len(self.dates)) + " dates used :\n")
            for date in self.dates:
                print(date, end = " | ")
                
        if hasattr(self, "last_computed_anomaly"):
            print("\n\n" + " Last computed anomaly : \n")
            print(self.last_computed_anomaly)
        else:
            print("Anomalies not computed")
    

    def delete_dirs(self,*key_paths):
        """
        Using keys to paths (usually added through add_path or add_dirpath), deletes directory containing the file, or the directory if the path already links to a directory. 

        Parameters
        ----------
        key_path : str
            Key in the dictionnary containing paths

        Returns
        -------
        None.

        """
        for key_path in key_paths:
            if key_path in self.paths:
                if self.paths[key_path].is_dir():
                    shutil.rmtree(self.paths[key_path])
                elif self.paths[key_path].is_file():
                    shutil.rmtree(self.paths[key_path].parent)
                      
   
    def delete_files(self,*key_paths):
        """
        Using keys to paths (usually added through add_path), deletes file 

        Parameters
        ----------
        key_path : str
            Key in the dictionnary containing paths

        Returns
        -------
        None.

        """
        for key_path in key_paths:
            if key_path in self.paths:
                if self.paths[key_path].is_file():
                    self.paths[key_path].unlink()
                 
    def delete_attributes(self,*attrs):
        """
        Using keys to attributes of the TileInfo object, deletes those attributes if they exist.

        Parameters
        ----------
        key_path : str
            Key of attributes of the object

        Returns
        -------
        None.

        """
        for attr in attrs:
            if hasattr(self, attr): delattr(self, attr)
                    
    def getdict_datepaths(self, key, path_dir):
        """
        Parameters
        ----------
        path_dir : str
            Directory containing files with filenames containing dates in the format YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, DD-MM-YYYY, DD_MM_YYYY or DDMMYYYY
    
        Returns
        -------
        dict_datepaths : dict
            Dictionnary linking formatted dates with the paths of the files from which the dates where extracted
    
        """
        path_dir=Path(path_dir)
        dict_datepaths={}
        for path in path_dir.glob("*"):
            if not(".xml" in str(path)): #To ignore temporary files
                formatted_date=retrieve_date_from_string(path.stem)
                if formatted_date != None: #To ignore files or directories with no dates which might be in the same directory
                    dict_datepaths[formatted_date] = path
            
        sorted_dict_datepaths = dict(sorted(dict_datepaths.items(), key=lambda key_value: key_value[0]))
        self.paths[key] = sorted_dict_datepaths
        
    def getdict_paths(self,
                      path_vi, path_masks):
        """
        Adds paths to vegetation index files and mask files to TileInfo object, along with the list of dates.

        Parameters
        ----------
        path_vi : str
            Directory containing vegetation index files with filenames containing dates in the format YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, DD-MM-YYYY, DD_MM_YYYY or DDMMYYYY
        path_masks : str
            Directory containing mask files with filenames containing dates in the format YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, DD-MM-YYYY, DD_MM_YYYY or DDMMYYYY


        """
        
        
        self.getdict_datepaths("VegetationIndex",path_vi)
        self.getdict_datepaths("Masks",path_masks)
        self.dates = np.array(list(self.paths["VegetationIndex"].keys()))
            

    def add_parameters(self, parameters):
        """
        Adds attribute 'parameters' to TileInfo object which contains dictionnary of parameters and their values.
        If attribute parameters already exists, checks for conflicts then updates parameters
        In case of conflicts, meaning if parameter was unknown or changed, the parameter 'Overwrite' is set to True and can be used to know when to deleted previous computation results.

        Parameters
        ----------
        parameters : dict
            Dictionnary containing parameters and their values

        """
        if not(hasattr(self, 'parameters')):
            self.parameters = parameters
            self.parameters["Overwrite"]=False
        else:
            self.parameters["Overwrite"]=False
            for parameter in parameters:
                if parameter in self.parameters:
                    if self.parameters[parameter]!=parameters[parameter]: #If parameter was changed
                        self.parameters["Overwrite"]=True
                else:#If unknown parameters
                    self.parameters["Overwrite"]=True
            self.parameters.update(parameters)
            
    def add_path(self, key, path):
        """
        Adds path to TileInfo object and creates parent directories if they don't exist already
        Path can then by retrieved with self.paths[key] where self is the TileInfo object name.

        Parameters
        ----------
        key : str
            Key for the paths dictionnary.
        path : str
            Path to add to the dictionnary.
        """
        
        #Transform to Path if not done already
        path=Path(path)
        #Creates missing directories
        path.parent.mkdir(parents=True, exist_ok=True)    
        #Saves paths in the object
        self.paths[key] = path
        
    def add_dirpath(self, key, path):
        """
        Adds path to a directory to TileInfo object and creates parent directories if they don't exist already
        Path can then by retrieved with self.paths[key] where self is the TileInfo object name.

        Parameters
        ----------
        key : str
            Key for the paths dictionnary.
        path : str
            Path to add to the dictionnary.
        """
        #Transform to WindowsPath if not done already
        path=Path(path)
        #Creates missing directories
        path.mkdir(parents=True, exist_ok=True)
        #Saves paths in the object
        self.paths[key] = path


    def search_new_dates(self):
        """
        Checks if there are new dates in vegetation index and mask directories, adds the paths and list of dates to the TileInfo object.

        """
        
        self.getdict_datepaths("VegetationIndex",self.paths["VegetationIndexDir"])
        self.getdict_datepaths("Masks",self.paths["MaskDir"])
        
        new_dates = np.array([date for date in self.paths["VegetationIndex"] if (not(hasattr(self, "dates")) or date > self.dates[-1])])
        self.dates = np.concatenate((self.dates, new_dates)) if hasattr(self, "dates") else new_dates
        
        # if hasattr(self, 'dates'):
        #     self.dates = np.array(list(self.paths["VegetationIndex"].keys())) >
        # self.dates = np.array(list(self.paths["VegetationIndex"].keys()))

        
        
def get_cloudiness(path_cloudiness, dict_path_bands, sentinel_source):
    """
    Imports, computes and stores cloudiness for all dates
    
    Parameters
    ----------
    path_cloudiness : str
        Path where the TileInfo object storing cloudiness information for each date is saved and imported from.
    dict_path_bands : dict
        Dictionnary where keys are dates, values are another dictionnary where keys are bands and values are their paths (dict_path_bands["YYYY-MM-DD"]["Mask"] -> Path to the mask)
    sentinel_source : str
        'THEIA', 'Scihub' or 'PEPS'

    Returns
    -------
    dict
        Dictionnary where keys are dates and values the cloudiness percentage

    """
    
    path_cloudiness= Path(path_cloudiness)
    cloudiness = TileInfo(path_cloudiness.parent)
    if path_cloudiness.exists():
        try:
            cloudiness=cloudiness.import_info(path_cloudiness)
        except:
            path_cloudiness.unlink()
    if not(hasattr(cloudiness, 'perc_cloud')):
        cloudiness.perc_cloud={}
    for date in dict_path_bands:
        if not(date in cloudiness.perc_cloud):
            cloudiness.perc_cloud[date] = get_date_cloudiness_perc(dict_path_bands[date], sentinel_source)
            
    cloudiness.save_info(path_cloudiness)
    return cloudiness.perc_cloud

def get_date_cloudiness_perc(date_paths, sentinel_source):
    """
    Computes cloudiness percentage of a Sentinel-2 date from the source mask (THEIA CLM or PEPS, scihub SCL)
    A 20m resolution band is necessary for THEIA data to determine swath cover. B11 is used but could be replaced with another 20m band.
    For THEIA, all pixels different to 0 in the mask are considered cloudy
    For Scihub and PEPS, all pixels different to 4 or 5 in the mask are considered cloudy
    
    Parameters
    ----------
    date_paths : Dictionnary where keys are bands and values are their paths
        DESCRIPTION.
    sentinel_source : TYPE
        'THEIA', 'Scihub' or 'PEPS'

    Returns
    -------
    float
        Cloudiness percentage

    """
    

    if sentinel_source=="THEIA":
        tile_mask_info = xr.Dataset({"mask": xr.open_rasterio(date_paths["Mask"]),
                                     "swath_cover": xr.open_rasterio(date_paths["B11"])})
        NbPixels = (tile_mask_info["swath_cover"]!=-10000).sum()
        NbCloudyPixels = (tile_mask_info["mask"]>0).sum()

    elif sentinel_source=="Scihub" or sentinel_source=="PEPS":
        cloud_mask = xr.open_rasterio(date_paths["Mask"])
        NbPixels = (cloud_mask!=0).sum()
        NbCloudyPixels = (~cloud_mask.isin([4,5])).sum()
    
    if NbPixels==0: #If outside of satellite swath
        return 2.0
    else:
        return float(NbCloudyPixels/NbPixels) #Number of cloudy pixels divided by number of pixels in the satellite swath



def get_raster_metadata(raster_path = None,raster = None, extent_shape_path = None):
    """
    From a raster path or a raster, extracts all metadata and returns it in a dictionnary. If extent_shape_path is given, the metadata from the raster clipped with the shape is returned

    Parameters
    ----------
    raster_path : str, optional
        path of a raster. The default is None.
    raster : xarray DataArray, optional
        xarray DataArray opened with xr.open_rasterio. The default is None.
    extent_shape_path : str, optional
        Path to a shapefile with a single polygon. The default is None.

    Returns
    -------
    raster_meta : dict
        Dictionnary containing all metadata (dims, coords, attrs, sizes, shape, extent).

    """
    if raster_path != None:
        raster = xr.open_rasterio(raster_path)
    raster.attrs["crs"] = raster.attrs["crs"].replace("+init=","")
    if extent_shape_path is not None:
        
        extent_shape = gp.read_file(extent_shape_path)
        
        extent_shape = extent_shape.to_crs(raster.crs)
        extent = extent_shape.total_bounds
        raster = raster.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]
        
    raster_meta = {"dims" : raster.dims,
                   "coords" : raster.coords,
                   "attrs" : raster.attrs,
                   "sizes" : raster.sizes,
                   "shape" : raster.shape,
                   "extent" : np.array([float(raster[dict(x=0,y=0)].coords["x"])-raster.attrs["transform"][0]/2,
                                        float(raster[dict(x=-1,y=-1)].coords["y"])-raster.attrs["transform"][0]/2,
                                        float(raster[dict(x=-1,y=-1)].coords["x"])+raster.attrs["transform"][0]/2,
                                        float(raster[dict(x=0,y=0)].coords["y"])+raster.attrs["transform"][0]/2])}
    
    raster_meta["attrs"]["transform"] = (raster_meta["attrs"]["transform"][0],
                                         raster_meta["attrs"]["transform"][1],
                                         raster_meta["extent"][0],
                                         raster_meta["attrs"]["transform"][3],
                                         raster_meta["attrs"]["transform"][4],
                                         raster_meta["extent"][3])

    return raster_meta
    

def clip_xarray(array, extent):
    """
    Clips xarray with x,y coordinates to an extent.

    Parameters
    ----------
    band_paths : xarray DataArray
        DataArray with x and y coordinates
    extent : list or 1D array, optional
        Extent used for cropping [xmin,ymin, xmax,ymax]. If None, there is no cropping. The default is None.

    Returns
    -------
    xarray DataArray
        DataArray clipped to the given extent

    """
    return array.loc[dict(x=slice(extent[0], extent[2]),y = slice(extent[3],extent[1]))]

def import_resampled_sen_stack(band_paths, list_bands, interpolation_order = 0, extent = None):
    """
    Imports and resamples the bands as an xarray

    Parameters
    ----------
    band_paths : dict
        Dictionnary where keys are bands and values are their paths
    list_bands : list
        List of bands to be imported
    interpolation_order : int, optional
        Order of interpolation as used in scipy's ndimage.zoom (0 = nearest neighbour, 1 = linear, 2 = bi-linear, 3 = cubic). The default is 0.
    extent : list or 1D array, optional
        Extent used for cropping [xmin,ymin, xmax,ymax]. If None, there is no cropping. The default is None.

    Returns
    -------
    concatenated_stack_bands : xarray
        3D xarray with dimensions x,y and band

    """
    #Importing data from files
    if extent is None:
            stack_bands = [xr.open_rasterio(band_paths[band]) for band in list_bands]
    else:
        stack_bands = [xr.open_rasterio(band_paths[band],chunks = 1280).loc[dict(x=slice(extent[0]-20, extent[2]+20),y = slice(extent[3]+20,extent[1]-20))].compute() for band in list_bands]
    
    crs = stack_bands[0].crs.replace("+init=","")
    #Resampling at 10m resolution
    for band_index in range(len(stack_bands)):
        if stack_bands[band_index].attrs["res"]==(20.0,20.0):            
            # stack_bands[band_index] = xr.DataArray(ndimage.zoom(stack_bands[band_index],zoom=[1,2.0,2.0],order=interpolation_order), 
            #                                        coords=stack_bands[0].coords)
            stack_bands[band_index] = xr.DataArray(ndimage.zoom(stack_bands[band_index],zoom=[1,2.0,2.0],order=interpolation_order), 
                                                   coords={"band" : [1], 
                                                           "y" : np.linspace(stack_bands[band_index].isel(x=0,y=0).y+5, stack_bands[band_index].isel(x=0,y=stack_bands[band_index].sizes["y"]-1).y-5, num=stack_bands[band_index].sizes["y"]*2),
                                                           "x" : np.linspace(stack_bands[band_index].isel(x=0,y=0).x-5, stack_bands[band_index].isel(x=stack_bands[band_index].sizes["x"]-1,y=0).x+5, num=stack_bands[band_index].sizes["x"]*2)},
                                                   dims=["band","y","x"])
        if extent is not None:
            stack_bands[band_index] = clip_xarray(stack_bands[band_index], extent)

    concatenated_stack_bands= xr.concat(stack_bands,dim="band")
    concatenated_stack_bands.coords["band"] = list_bands
    # concatenated_stack_bands=concatenated_stack_bands.chunk({"band": 1,"x" : -1,"y" : 100})
    concatenated_stack_bands.attrs["nodata"] = 0
    concatenated_stack_bands.attrs["crs"]=crs
    return concatenated_stack_bands


        
def import_forest_mask(forest_mask_path,chunks = None):
    """
    Imports forest mask

    Parameters
    ----------
    forest_mask_path : str
        Path of the forest mask binary raster.
    chunks : int, optional
        Chunks for import as dask array. If None, data is imported as xarray. The default is None.

    Returns
    -------
    xarray DataArray
        Binary array containing True if pixels are inside the region of interest.

    """
    forest_mask = xr.open_rasterio(forest_mask_path,chunks = chunks).squeeze("band")
    # forest_mask=forest_mask.rename({"band" : "Mask"})
    return forest_mask.astype(bool)


def import_stackedmaskedVI(tuile,min_date = None, max_date=None,chunks = None):
    """
    Imports 3D arrays of the vegetation index series and masks

    Parameters
    ----------
    tuile : Object of class TileInfo
        Object containing paths of vegetation index and masks for each date
    max_date : str, optional
        Date in the format "YYYY-MM-DD". Only dates anterior to max_date are imported. If None, all dates are imported. The default is None.
    chunks : int, optional
        Chunks for import as dask array. If None, data is imported as xarray. The default is None.

    Returns
    -------
    stack_vi : xarray.DataArray or dask array
        DataArray containing vegetation index value with dimension Time, x and y
    stack_masks : xarray.DataArray or dask array
        DataArray containing mask value with dimension Time, x and y

    """
    
    if max_date is None and min_date is None:
        dates = [date for date in tuile.paths["VegetationIndex"]]
    elif max_date is None and min_date is not None:
        dates = [date for date in tuile.paths["VegetationIndex"] if date >= min_date]
    elif max_date is not None and min_date is None:
        dates = [date for date in tuile.paths["VegetationIndex"] if date <= max_date]
    else:
        dates = [date for date in tuile.paths["VegetationIndex"]if date >= min_date & date <= max_date]
        
    list_vi=[xr.open_rasterio(tuile.paths["VegetationIndex"][date],chunks =chunks) for date in dates]
    stack_vi=xr.concat(list_vi,dim="Time")
    stack_vi=stack_vi.assign_coords(Time=dates)
    stack_vi=stack_vi.squeeze("band")
    stack_vi=stack_vi.chunk({"Time": -1,"x" : chunks,"y" : chunks})    
    # stack_vi["DateNumber"] = ("Time", np.array([(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days for date in np.array(stack_vi["Time"])]))

    
    list_mask=[xr.open_rasterio(tuile.paths["Masks"][date],chunks =chunks) for date in dates]
    stack_masks=xr.concat(list_mask,dim="Time")
    stack_masks=stack_masks.assign_coords(Time=dates).astype(bool)
    stack_masks=stack_masks.squeeze("band")
    stack_masks=stack_masks.chunk({"Time": -1,"x" : chunks,"y" : chunks})
    # stack_masks["DateNumber"] = ("Time", np.array([(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days for date in np.array(stack_masks["Time"])]))

    return stack_vi, stack_masks

    
def import_coeff_model(path, chunks = None):
    """
    Imports array containing the coefficients to the model for vegetation index prediction. The array has a "coeff" dimension containing each coefficient.

    Parameters
    ----------
    path : str
        ath of the file.
    chunks : TYPE, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    coeff_model : xarray DataArray of dask array
        Array containing the coefficients to the model for vegetation index prediction.

    """
    
    coeff_model = xr.open_rasterio(path,chunks = chunks)
    coeff_model = coeff_model.rename({"band": "coeff"})
    return coeff_model

def import_first_detection_date_index(path,chunks = None):
    """
    Imports array containing the index of the first date used for detection instead of training

    Parameters
    ----------
    path : str
        Path of the file.
    chunks : int, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    first_detection_date_index : xarray DataArray of dask array
        Array containing the index of the first date used for detection instead of training

    """
    
    first_detection_date_index=xr.open_rasterio(path,chunks = chunks).squeeze("band")
    return first_detection_date_index

def import_dieback_data(dict_paths, chunks = None):
    """
    Imports data relating to dieback detection

    Parameters
    ----------
    dict_paths : dict
        Dictionnary containg the keys "state_dieback", "first_date_dieback" and "count_dieback" whose values are the paths to the corresponding dieback data file.
    chunks : int, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    dieback_data : xarray DataSet or dask DataSet
        DataSet containing three DataArrays, "state" containing the state of the pixel after computations, "first_date" containing the index of the date of the first anomaly, "count" containing the number of successive anomalies if "state" is True, or conversely the number of successive dates without anomalies. 

    """
    
    state_dieback = xr.open_rasterio(dict_paths["state_dieback"],chunks = chunks).astype(bool)
    first_date_dieback = xr.open_rasterio(dict_paths["first_date_dieback"],chunks = chunks)
    count_dieback = xr.open_rasterio(dict_paths["count_dieback"],chunks = chunks)
    
    dieback_data=xr.Dataset({"state": state_dieback,
                     "first_date": first_date_dieback,
                     "count" : count_dieback})
    dieback_data=dieback_data.squeeze("band")

    return dieback_data
        
def initialize_dieback_data(shape,coords):
    """
    Initializes data relating to dieback detection

    Parameters
    ----------
    shape : tuple
        Tuple with sizes for the resulting array 
    coords : Coordinates attribute of xarray DataArray
        Coordinates y and x

    Returns
    -------
    dieback_data : xarray DataSet or dask DataSet
        DataSet containing three DataArrays, "state" containing the state of the pixel after computations, "first_date" containing the index of the date of the first anomaly, "count" containing the number of successive anomalies if "state" is True, or conversely the number of successive dates without anomalies. 
        For all three arrays, all pixels are intitialized at zero.


    """
    
    count_dieback= np.zeros(shape,dtype=np.uint8) #np.int8 possible ?
    first_date_dieback=np.zeros(shape,dtype=np.uint16) #np.int8 possible ?
    state_dieback=np.zeros(shape,dtype=bool)
    
    dieback_data=xr.Dataset({"state": xr.DataArray(state_dieback, coords=coords),
                         "first_date": xr.DataArray(first_date_dieback, coords=coords),
                         "count" : xr.DataArray(count_dieback, coords=coords)})
    
    return dieback_data



def import_soil_data(dict_paths, chunks = None):
    """
    Imports data relating to soil detection

    Parameters
    ----------
    dict_paths : dict
        Dictionnary containg the keys "state_soil", "first_date_soil" and "count_soil" whose values are the paths to the corresponding soil detection data file.
    chunks : int, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    soil_data : xarray DataSet or dask DataSet
        DataSet containing three DataArrays, "state" containing the state of the pixel after computations (True for soil), "first_date" containing the index of the date of the first soil anomaly, "count" containing the number of successive soil anomalies.

    """
    
    state_soil = xr.open_rasterio(dict_paths["state_soil"], chunks = chunks).astype(bool)
    first_date_soil = xr.open_rasterio(dict_paths["first_date_soil"], chunks = chunks)
    count_soil = xr.open_rasterio(dict_paths["count_soil"], chunks = chunks)
    
    soil_data=xr.Dataset({"state": state_soil,
                     "first_date": first_date_soil,
                     "count" : count_soil})
    soil_data=soil_data.squeeze("band")

    return soil_data

def initialize_soil_data(shape,coords):
    """
    Initializes data relating to soil detection

    Parameters
    ----------
    shape : tuple
        Tuple with sizes for the resulting array 
    coords : Coordinates attribute of xarray DataArray
        Coordinates y and x

    Returns
    -------
    soil_data : xarray DataSet or dask DataSet
        DataSet containing three DataArrays, "state" containing the state of the pixel after computations, "first_date" containing the index of the date of the first soil anomaly, "count" containing the number of successive soil anomalies
        For all three arrays, all pixels are intitialized at zero.


    """
    state_soil=np.zeros(shape,dtype=bool)
    first_date_soil=np.zeros(shape,dtype=np.uint16)
    count_soil= np.zeros(shape,dtype=np.uint16)
    
    soil_data=xr.Dataset({"state": xr.DataArray(state_soil, coords=coords),
                         "first_date": xr.DataArray(first_date_soil, coords=coords),
                         "count" : xr.DataArray(count_soil, coords=coords)}).squeeze("band")
    return soil_data

def initialize_confidence_data(shape,coords):
    """
    Initializes data relating to confidence index

    Parameters
    ----------
    shape : tuple
        Tuple with sizes for the resulting array 
    coords : Coordinates attribute of xarray DataArray
        Coordinates y and x

    Returns
    -------
    nb_dates : xarray DataArray (x,y)
        Number of unmasked sentinel dates since the first anomaly for each pixel.
    sum_diff : xarray DataArray (x,y)
        Cumulative sum of differences between the vegetation index and its prediction for each date.

    """
    
    
    nb_dates=xr.DataArray(np.zeros(shape,dtype=np.uint16), coords=coords)
    sum_diff=xr.DataArray(np.zeros(shape,dtype=np.float), coords=coords)

    return nb_dates, sum_diff

def import_confidence_data(dict_paths, chunks = None):
    """
    Imports data relating to confidence index

    Parameters
    ----------
    dict_paths : dict
        Dictionnary containing keys "confidence_index" and "nb_dates" whose values are the paths to the rasters.
    Returns
    -------
    confidence_index : xarray DataArray (x,y)
        Confidence index
    nb_dates : xarray DataArray (x,y)
        Number of unmasked sentinel dates since the first anomaly for each pixel.

    """
    confidence_index=xr.open_rasterio(dict_paths["confidence_index"], chunks = chunks).squeeze("band")
    nb_dates=xr.open_rasterio(dict_paths["nb_dates"], chunks = chunks).squeeze("band")

    return confidence_index, nb_dates

def import_masked_vi(dict_paths, date, chunks = None):
    """
    Imports masked vegetation index

    Parameters
    ----------
    dict_paths : str
        Dictionnary where key "VegetationIndex" returns a dictionnary where keys are SENTINEL dates and values are paths to the files containing the values of the vegetation index for the SENTINEL date, and key "Masks" returns the equivalent for the masks files.
    date : str
        Date in the format "YYYY-MM-DD"
    chunks : int, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    masked_vi : xarray DataSet
        DataSet containing two DataArrays, "vegetation_index" containing vegetation index values, and "mask" containing mask values.

    """
    
    vegetation_index = xr.open_rasterio(dict_paths["VegetationIndex"][date],chunks = chunks)
    mask=xr.open_rasterio(dict_paths["Masks"][date],chunks = chunks).astype(bool)
    
    masked_vi=xr.Dataset({"vegetation_index": vegetation_index,
                             "mask": mask})
    masked_vi=masked_vi.squeeze("band")

    return masked_vi



def import_stacked_anomalies(paths_anomalies, chunks = None):
    """
    Imports all stacked anomalies

    Parameters
    ----------
    dict_paths : str
        Dictionnary where keys are dates in the format "YYYY-MM-DD" and values are the paths to the raster file containing anomaly data.
    chunks : int, optional
        Chunk size for import as dask array. The default is None.

    Returns
    -------
    stack_anomalies : xarray DataArray
        3D binary DataArray with value True where there are anomalies, with Time coordinates.

    """
    list_anomalies=[xr.open_rasterio(paths_anomalies[date], chunks = chunks) for date in paths_anomalies]
    stack_anomalies=xr.concat(list_anomalies,dim="Time")
    stack_anomalies=stack_anomalies.assign_coords(Time=[date for date in paths_anomalies.keys()])
    stack_anomalies=stack_anomalies.squeeze("band")
    stack_anomalies=stack_anomalies.chunk({"Time": -1,"x" : chunks,"y" : chunks})
    # stack_anomalies["DateNumber"] = ("Time", np.array([(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days for date in np.array(stack_anomalies["Time"])]))
    return stack_anomalies.astype(bool)



    