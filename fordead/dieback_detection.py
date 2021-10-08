# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:34:34 2020

@author: Raphael Dutrieux
"""

from fordead.masking_vi import get_dict_vi
import xarray as xr
import numpy as np

def detection_anomalies(masked_vi, predicted_vi, threshold_anomaly, vi, path_dict_vi = None):
    """
    Detects anomalies by comparison between predicted and calculated vegetation index. The array returns contains True where the difference between the vegetation index and its prediction is above the threshold in the direction of the specified dieback change direction of the vegetation index. 

    Parameters
    ----------
    masked_vi : xarray Dataset
        Dataset containing two DataArrays, "vegetation_index" containing the vegetation index values, and "mask" containing the mask values (True if masked)
    predicted_vi : array (x,y)
        Array containing the vegetation index predicted by the model
    threshold_anomaly : float
        Threshold used to compare predicted and calculated vegetation index.
    vi : str
        Name of the used vegetation index
    path_dict_vi : str, optional
        Path to a text file containing vegetation indices information, where is indicated whether the index rises of falls in case of forest dieback. See get_dict_vi documentation. The default is None.


    Returns
    -------
    anomalies : array (x,y) (bool)
        Array, pixel value is True if an anomaly is detected.

    """

    dict_vi = get_dict_vi(path_dict_vi)
    
    if dict_vi[vi]["dieback_change_direction"] == "+":
        diff_vi = (masked_vi["vegetation_index"]-predicted_vi)*(~masked_vi["mask"])
    elif dict_vi[vi]["dieback_change_direction"] == "-":
        diff_vi = (predicted_vi - masked_vi["vegetation_index"])*(~masked_vi["mask"])
    else:
        raise Exception("Unrecognized dieback_change_direction in " + path_dict_vi + " for vegetation index " + vi)
    
    anomalies = diff_vi > threshold_anomaly
    
    return anomalies.squeeze("Time"), diff_vi.squeeze("Time")

def detection_dieback(dieback_data, anomalies, mask, date_index):
    """
    Updates dieback data using anomalies. Successive anomalies are counted for pixels considered healthy, and successive dates without anomalies are counted for pixels considered suffering from dieback. The state of the pixel changes when the count reaches 3.
    
    Parameters
    ----------
    dieback_data : Dataset with three arrays : 
        "count" which is the number of successive anomalies, 
        "state" which is True where pixels are detected as suffering from dieback, False where they are considered healthy.
        "first date" which contains the index of the date of the first anomaly.
    anomalies : array (x,y) (bool)
        Array, pixel value is True if an anomaly is detected.
    mask : array (x,y) (bool)
        Array, True where pixels are masked
    date_index : int
        Index of the date

    Returns
    -------
    dieback_data : Dataset
        Dataset with the three arrays updated with the results from the date being analysed

    """

    dieback_data["count"] = xr.where(~mask & (anomalies!=dieback_data["state"]),dieback_data["count"]+1,dieback_data["count"])
    dieback_data["count"] = xr.where(~mask & (anomalies==dieback_data["state"]),0,dieback_data["count"])
    changing_pixels = dieback_data["count"]==3

    dieback_data["state"] = xr.where(changing_pixels, ~dieback_data["state"], dieback_data["state"]) #Changement d'état si CompteurScolyte = 3 et date valide
    dieback_data["first_date"] = dieback_data["first_date"].where(~changing_pixels,dieback_data["first_date_unconfirmed"])
    dieback_data["count"] = xr.where(changing_pixels, 0,dieback_data["count"])
    dieback_data["first_date_unconfirmed"]=xr.where(~mask & (dieback_data["count"]==1), date_index, dieback_data["first_date_unconfirmed"]) #Garde la première date de détection de scolyte sauf si déjà détécté comme scolyte

    return dieback_data,changing_pixels

def save_stress(stress_data, dieback_data, changing_pixels, diff_vi):
    stress_data["nb_periods"]=stress_data["nb_periods"]+changing_pixels*(~dieback_data["state"]) #Adds one to the number of stress periods when pixels change back to normal
#AVEC MODULO   
    # nb_periods_modulo = np.mod(stress_data["nb_periods"],stress_data.sizes["period"])

    # relevant_period = stress_data["cum_diff"]["period"] != (nb_periods_modulo+1) # +1 because periods coordinates start at 1
    # potential_stressed_pixels = (dieback_data["count"]==0) & ~dieback_data["state"]
    
    # stress_data["cum_diff"] = stress_data["cum_diff"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["cum_diff"]+diff_vi))
    # stress_data["nb_dates"] = stress_data["nb_dates"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["nb_dates"]+1))
    
    # nb_changes = np.mod(stress_data["nb_periods"]*2+dieback_data["state"],stress_data.sizes["period"]*2+1) #Number of the change
    # stress_data["date"] = stress_data["date"].where(~changing_pixels | (stress_data["date"]["change"] != nb_changes), dieback_data["first_date"])

#SANS MODULO    

    relevant_period = stress_data["cum_diff"]["period"] != (stress_data["nb_periods"]+1) #ajouter modulo +1 because periods coordinates start at 1
    potential_stressed_pixels = (dieback_data["count"]==0) & ~dieback_data["state"]
    
    stress_data["cum_diff"] = stress_data["cum_diff"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["cum_diff"]+diff_vi))
    stress_data["nb_dates"] = stress_data["nb_dates"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["nb_dates"]+1))
    
    nb_changes = stress_data["nb_periods"]*2+dieback_data["state"] #Number of the change #ajouter modulo
    stress_data["date"] = stress_data["date"].where(~changing_pixels | (stress_data["date"]["change"] != nb_changes), dieback_data["first_date"])
    return stress_data
    
    
