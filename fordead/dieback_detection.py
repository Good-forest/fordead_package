# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:34:34 2020

@author: Raphael Dutrieux
"""

from fordead.masking_vi import get_dict_vi
import numpy as np
import xarray as xr

def detection_anomalies(vegetation_index, mask, predicted_vi, threshold_anomaly, vi, path_dict_vi = None):
    """
    Detects anomalies by comparison between predicted and calculated vegetation index. The array returns contains True where the difference between the vegetation index and its prediction is above the threshold in the direction of the specified dieback change direction of the vegetation index. 

    Parameters
    ----------
    vegetation_index : xarray DataArray
        DataArray containing uncorrected vegetation index values
    mask : xarray DataArray
        DataArray containing mask values.
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
    diff_vi : array (x,y) (float)
        Array containing the difference between the vegetation index and its prediction

    """

    dict_vi = get_dict_vi(path_dict_vi)
    
    if dict_vi[vi]["dieback_change_direction"] == "+":
        diff_vi = (vegetation_index-predicted_vi)*(~mask)
    elif dict_vi[vi]["dieback_change_direction"] == "-":
        diff_vi = (predicted_vi - vegetation_index)*(~mask)
    else:
        raise Exception("Unrecognized dieback_change_direction in " + path_dict_vi + " for vegetation index " + vi)
    
    anomalies = diff_vi > threshold_anomaly
    
    return anomalies.squeeze("Time").squeeze("band"), diff_vi.squeeze("Time").squeeze("band") #.squeeze("Time").squeeze("band")

def detection_dieback(dieback_data, anomalies, mask, date_index, date):
    """
    Updates dieback data using anomalies. Successive anomalies are counted for pixels considered healthy, and successive dates without anomalies are counted for pixels considered suffering from dieback. The state of the pixel changes when the count reaches 3.
    
    Parameters
    ----------
    dieback_data : Dataset with four arrays : 
        "count" which is the number of successive anomalies, 
        "state" which is True where pixels are detected as suffering from dieback, False where they are considered healthy.
        "first_date" contains the index of the date of the first anomaly then confirmed
        "first_date_unconfirmed" containing the date of pixel change, first anomaly if pixel is not detected as dieback, first non-anomaly if pixel is detected as dieback
    anomalies : array (x,y) (bool)
        Array, pixel value is True if an anomaly is detected.
    mask : array (x,y) (bool)
        Array, True where pixels are masked
    date_index : int
        Index of the date

    Returns
    -------
    dieback_data : Dataset
        Dataset with the four arrays updated with the results from the date being analysed
    changing_pixels : DataArray
        Binary array (x,y) containing True where pixels change state is confirmed with a third successive anomaly
    """
    dieback_data["count"] = xr.where(~mask & (anomalies != dieback_data["state"]), dieback_data["count"] + 1 , dieback_data["count"])
    dieback_data["count"] = xr.where(~mask & (anomalies == dieback_data["state"]), 0, dieback_data["count"])

    delay_since_first_date_unconfirmed = (date - dieback_data["first_date_unconfirmed_date"]) / 86400
    # delay_more_than_30_days = delay_since_first_date_unconfirmed > 30
    # last_duration_less_than_90_days = dieback_data["last_duration"] < 90
    # changing_pixels = (dieback_data["count"] == 3) & (
    #     (dieback_data["state"] != True) |
    #     (delay_more_than_30_days & last_duration_less_than_90_days)
    # )
    changing_pixels = (dieback_data["count"] == 3) | ((dieback_data["count"] == 2) & (dieback_data["state"] == False) & (delay_since_first_date_unconfirmed > 15))
    # changing_pixels = dieback_data["count"] == 3

    #4. State = ~State si Changement de pixel
    dieback_data["state"] = xr.where(changing_pixels, ~dieback_data["state"], dieback_data["state"])

    #5. first_date = first_date_unconfirmed si Changement de pixel
    dieback_data["first_date"] = dieback_data["first_date"].where(~changing_pixels, dieback_data["first_date_unconfirmed"])

    # 6. count = 0 si changement de pixel
    dieback_data["count"] = xr.where(changing_pixels, 0, dieback_data["count"])

    epoch = np.datetime64("1970-01-01", 's').astype(float)
    dieback_data["last_duration"] = xr.where(~mask & (dieback_data["count"] == 1) & (dieback_data["first_date_unconfirmed_date"] > epoch), dieback_data["first_date_unconfirmed_date"] - date, dieback_data["last_duration"])

    # 7. first_date_unconfirmed = date_index si premiere anomalie
    dieback_data["first_date_unconfirmed"] = xr.where(~mask & (dieback_data["count"] == 1), date_index, dieback_data["first_date_unconfirmed"])

    dieback_data["first_date_unconfirmed_date"] = xr.where(~mask & (dieback_data["count"] == 1), date, dieback_data["first_date_unconfirmed_date"])

    return dieback_data,changing_pixels

def save_stress(stress_data, dieback_data, changing_pixels, diff_vi, mask, stress_index_mode):
    """
    Updates stress data, saves the date of pixel state changes,
    adds one to the number of stress periods when pixels change back to normal,
    adds the difference between vegetation index and its prediction multiplied by
    the weight of the date if "stress_index_mode" is "weighted_mean",
    iterates the number of dates in the stress periods for unmasked pixels.

    Parameters
    ----------
    stress_data : xarray DataSet or dask DataSet
        DataSet containing four DataArrays, "date" containing the date index of each pixel state change, "nb_periods" containing the total number of stress periods detected for each pixel, "cum_diff" containing for each stress period the sum of the difference between the vegetation index and its prediction, multiplied by the weight if stress_index_mode is "weighted_mean", and "nb_dates" containing the number of valid dates of each stress period.
    dieback_data : Dataset with four arrays : 
        "count" which is the number of successive anomalies, 
        "state" which is True where pixels are detected as suffering from dieback, False where they are considered healthy.
        "first_date" contains the index of the date of the first anomaly then confirmed
        "first_date_unconfirmed" containing the date of pixel change, first anomaly if pixel is not detected as dieback, first non-anomaly if pixel is detected as dieback
    changing_pixels : DataArray
        Binary array (x,y) containing True where pixels change state is confirmed with a third successive anomaly
    diff_vi : array (x,y) (float)
        Array containing the difference between the vegetation index and its prediction
    mask : array (x,y) (bool)
        Array containing the mask values (True if masked)
    stress_index_mode : str
        Chosen stress index, if 'mean', diff_vi is simply added to "cum_diff", if 'weighted_mean', diff_vi is added to "cum_diff" after being multiplied with "nb_dates", the number of the date from the first anomaly.

    Returns
    -------
    stress_data : xarray DataSet or dask DataSet
        Updated dataSet containing four DataArrays, "date" containing the date index of each pixel state change, "nb_periods" containing the total number of stress periods detected for each pixel, "cum_diff" containing for each stress period the sum of the difference between the vegetation index and its prediction, multiplied by the weight if stress_index_mode is "weighted_mean", and "nb_dates" containing the number of valid dates of each stress period.


    """
    
    stress_data["nb_periods"]=stress_data["nb_periods"]+changing_pixels*(~dieback_data["state"]) #Adds one to the number of stress periods when pixels change back to normal

    relevant_period = stress_data["cum_diff"]["period"] != (stress_data["nb_periods"]+1) #+1 because periods coordinates start at 1
    potential_stressed_pixels = (dieback_data["count"]==0) & ~dieback_data["state"]
    stress_data["nb_dates"] = stress_data["nb_dates"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["nb_dates"]+~mask))

    if stress_index_mode == "mean":
        stress_data["cum_diff"] = stress_data["cum_diff"].where(relevant_period, xr.where(potential_stressed_pixels, 0, stress_data["cum_diff"]+diff_vi))
    elif stress_index_mode == "weighted_mean":
        stress_data["cum_diff"] = stress_data["cum_diff"].where(
            relevant_period, 
            xr.where(
                potential_stressed_pixels, 0, stress_data["cum_diff"]+
                diff_vi*stress_data["nb_dates"].sel(
                    period = (stress_data["nb_periods"]+1).where(
                        stress_data["nb_periods"]<=stress_data.sizes["period"]-1,
                        stress_data.sizes["period"]
                        )
                    )
                )
            ) #The selection probably makes no sense for pixels with nb_periods higher that stress_data.sizes["period"], but it doesn't matter since they are excluded from result exports, but it removes bugs of inexistant period values.
    else:
        raise Exception("Unrecognized stress_index_mode")
        
    nb_changes = stress_data["nb_periods"]*2+dieback_data["state"] #Number of the change
    stress_data["date"] = stress_data["date"].where(~changing_pixels | (stress_data["date"]["change"] != nb_changes), dieback_data["first_date"])
    return stress_data.squeeze("band")
    
    
