# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:42:31 2020

@author: admin
"""

import numpy as np
import xarray as xr
import re
import datetime
from pathlib import Path
# from path import Path

import pickle
import shutil

    
def get_band_paths(dict_sen_paths):
    DictSentinelPaths={}
    for date in dict_sen_paths:
        # AllPaths=glob(dict_sen_paths[date]+"/**",recursive=True)
        AllPaths = dict_sen_paths[date].glob("**/*")
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
            if ("_B8" in path or "_B08" in path) and not("_B8A" in path):
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
        Initialize TileInfo object, deletes previous results if they exist.
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)   
        self.paths={}
        
        # print(globals())
        #Deletes previous results if object already exists
        if (self.data_directory / "PathsInfo").exists():
            with open(self.data_directory / "PathsInfo", 'rb') as f:
                tuile2 = pickle.load(f)
                
            for key_path in tuile2.paths:
                if not(key_path in ["VegetationIndex","Masks","ForestMask", "used_area_mask"]):
                    if isinstance(tuile2.paths[key_path],type(self.data_directory)): #Check if value is a path
                        tuile2.delete_dir(key_path)
            print("Previous results detected and deleted")
            
    # def add_parameters(self,dict_param):
    #     if hasattr(self, 'property'):
    #         a.property
        
        
    def delete_dir(self,key_path):
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
        if key_path in self.paths:
            if self.paths[key_path].is_dir():
                shutil.rmtree(self.paths[key_path])
            elif self.paths[key_path].is_file():
                shutil.rmtree(self.paths[key_path].parent)
            
    def getdict_datepaths(self, key, path_dir):
        """
        Parameters
        ----------
        path_dir : pathlib.WindowsPath
            Directory containing files with filenames containing dates in the format YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, DD-MM-YYYY, DD_MM_YYYY or DDMMYYYY
    
        Returns
        -------
        dict_datepaths : dict
            Dictionnary linking formatted dates with the paths of the files from which the dates where extracted
    
        """
        dict_datepaths={}
        for path in path_dir.glob("*"):
            formatted_date=retrieve_date_from_string(path.stem)
            if formatted_date != None: #To ignore files or directories with no dates which might be in the same directory
                dict_datepaths[formatted_date] = path
            
        sorted_dict_datepaths = dict(sorted(dict_datepaths.items(), key=lambda key_value: key_value[0]))
        self.paths[key] = sorted_dict_datepaths
        
    def getdict_paths(self,
                      path_vi, path_masks, path_forestmask = None):
        
        self.getdict_datepaths("VegetationIndex",path_vi)
        self.getdict_datepaths("Masks",path_masks)
        self.paths["ForestMask"]=path_forestmask
        self.dates = np.array(list(self.paths["VegetationIndex"].keys()))
            
    def add_path(self, key, path):
        #Transform to WindowsPath if not done already
        path=Path(path)
        #Creates missing directories
        path.parent.mkdir(parents=True, exist_ok=True)    
        #Saves paths in the object
        self.paths[key] = path

    def add_dirpath(self, key, path):
        #Transform to WindowsPath if not done already
        path=Path(path)
        #Creates missing directories
        path.mkdir(parents=True, exist_ok=True)    
        #Saves paths in the object
        self.paths[key] = path

    def save_info(self, path= None):
        """
        Saves the TileInfo object in its data_directory by default or in specified location
        """
        if path==None:
            path=self.data_directory / "PathsInfo"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def search_new_dates(self):
        path_vi=self.paths["VegetationIndex"][self.dates[0]].parent
        path_masks=self.paths["Masks"][self.dates[0]].parent
        self.getdict_datepaths("VegetationIndex",path_vi)
        self.getdict_datepaths("Masks",path_masks)
        self.dates = np.array(list(self.paths["VegetationIndex"].keys()))
        


def import_forest_mask(PathMaskForet):
    forest_mask = xr.open_rasterio(PathMaskForet,chunks =1000)
    forest_mask=forest_mask[0,:,:]
    # forest_mask=forest_mask.rename({"band" : "Mask"})
    return forest_mask.astype(bool)


def import_stackedmaskedVI(tuile,date_lim_learning=None):
    """

    Parameters
    ----------
    tuile : Object of class TileInfo
        Object containing paths of vegetation index and masks for each date

    Returns
    -------
    stack_vi : xarray.DataArray
        DataArray containing vegetation index value with dimension Time, x and y
    stack_masks : xarray.DataArray
        DataArray containing mask value with dimension Time, x and y

    """
    if date_lim_learning==None:
        filter_dates=False
        date_lim_learning=""
    else:
        filter_dates=True
        
    list_vi=[xr.open_rasterio(tuile.paths["VegetationIndex"][date],chunks =1000) for date in tuile.paths["VegetationIndex"] if date <= date_lim_learning or not(filter_dates)]
    stack_vi=xr.concat(list_vi,dim="Time")
    stack_vi=stack_vi.assign_coords(Time=[date for date in tuile.paths["VegetationIndex"].keys() if date <= date_lim_learning or not(filter_dates)])
    stack_vi=stack_vi.sel(band=1)
    stack_vi=stack_vi.chunk({"Time": -1,"x" : 1000,"y" : 1000})    
    stack_vi["DateNumber"] = ("Time", np.array([(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-06-23', '%Y-%m-%d')).days for date in np.array(stack_vi["Time"])]))

    
    list_mask=[xr.open_rasterio(tuile.paths["Masks"][date],chunks =1000) for date in tuile.paths["Masks"] if date <= date_lim_learning or not(filter_dates)]
    stack_masks=xr.concat(list_mask,dim="Time")
    stack_masks=stack_masks.assign_coords(Time=[date for date in tuile.paths["Masks"].keys() if date <= date_lim_learning or not(filter_dates)]).astype(bool)
    stack_masks=stack_masks.sel(band=1)
    stack_masks=stack_masks.chunk({"Time": -1,"x" : 1000,"y" : 1000})
    stack_masks["DateNumber"] = ("Time", np.array([(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-06-23', '%Y-%m-%d')).days for date in np.array(stack_masks["Time"])]))
    return stack_vi, stack_masks

    
def import_coeff_model(path):
    coeff_model = xr.open_rasterio(path,chunks = 1000)
    return coeff_model

def import_last_training_date(path):
    last_training_date=xr.open_rasterio(path,chunks = 1000)
    last_training_date=last_training_date.sel(band=1)
    return last_training_date

def import_decline_data(dict_paths):
    state_decline = xr.open_rasterio(dict_paths["state_decline"],chunks = 1000).astype(bool)
    first_date_decline = xr.open_rasterio(dict_paths["first_date_decline"],chunks = 1000)
    count_decline = xr.open_rasterio(dict_paths["count_decline"],chunks = 1000)
    
    decline_data=xr.Dataset({"state": state_decline,
                     "first_date": first_date_decline,
                     "count" : count_decline})
    decline_data=decline_data.sel(band=1)

    return decline_data
        
def initialize_decline_data(shape,coords):
    
    count_decline= np.zeros(shape,dtype=np.uint8) #np.int8 possible ?
    first_date_decline=np.zeros(shape,dtype=np.uint16) #np.int8 possible ?
    state_decline=np.zeros(shape,dtype=bool)
    
    decline_data=xr.Dataset({"state": xr.DataArray(state_decline, coords=coords),
                         "first_date": xr.DataArray(first_date_decline, coords=coords),
                         "count" : xr.DataArray(count_decline, coords=coords)})
    
    return decline_data

def import_masked_vi(dict_paths, date):
    vegetation_index = xr.open_rasterio(dict_paths["VegetationIndex"][date],chunks = 1000)
    mask=xr.open_rasterio(dict_paths["Masks"][date],chunks = 1000).astype(bool)
    
    masked_vi=xr.Dataset({"vegetation_index": vegetation_index,
                             "mask": mask})
    masked_vi=masked_vi.sel(band=1)

    return masked_vi



# def ImportMaskForet(PathMaskForet):
#     MaskForet = xr.open_rasterio(PathMaskForet)
#     with rasterio.open(PathMaskForet) as BDFORET: 
#         RasterizedBDFORET = BDFORET.read(1).astype("bool")
#         profile = BDFORET.profile
#         CRS_Tuile = int(str(profile["crs"])[5:])
#     return RasterizedBDFORET,profile,CRS_Tuile