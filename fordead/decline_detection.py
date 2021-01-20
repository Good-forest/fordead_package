# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:34:34 2020

@author: Raphael Dutrieux
"""

import datetime
from fordead.ModelVegetationIndex import compute_HarmonicTerms
from fordead.masking_vi import get_dict_vi
import xarray as xr

import numpy as np

def prediction_vegetation_index(coeff_model,date_list):
    """
    Predicts the vegetation index from the model coefficients and the date
    
    Parameters
    ----------
    coeff_model : array (5,x,y)
        Array containing the five coefficients of the vegetation index model for each pixel
    date : str
        Date in the format "YYYY-MM-DD"

    Returns
    -------
    predicted_vi : array (x,y)
        Array containing predicted vegetation index from the model

    """
        # harmonic_terms = np.array([compute_HarmonicTerms(DateAsNumber) for DateAsNumber in stack_vi["DateNumber"]])
        # harmonic_terms = xr.DataArray(harmonic_terms, coords={"Time" : stack_vi["Time"], "band" : [1,2,3,4,5]},dims=["Time", "band"])
        # predicted_vi = sum(coeff_model * harmonic_terms)
        
    date_as_number_list=[(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-06-23', '%Y-%m-%d')).days for date in date_list]
    harmonic_terms = np.array([compute_HarmonicTerms(DateAsNumber) for DateAsNumber in date_as_number_list])
    harmonic_terms = xr.DataArray(harmonic_terms, coords={"Time" : date_list, "coeff" : [1,2,3,4,5]},dims=["Time", "coeff"])
    # harmonic_terms = compute_HarmonicTerms(date_as_number)
    # harmonic_terms = xr.DataArray(harmonic_terms, coords={"band" : [1,2,3,4,5]},dims=["band"])
    
    predicted_vi = sum(coeff_model * harmonic_terms)
    return predicted_vi

# def prediction_vegetation_index(coeff_model,date):
#     """
#     Predicts the vegetation index from the model coefficients and the date
    
#     Parameters
#     ----------
#     coeff_model : array (5,x,y)
#         Array containing the five coefficients of the vegetation index model for each pixel
#     date : str
#         Date in the format "YYYY-MM-DD"

#     Returns
#     -------
#     predicted_vi : array (x,y)
#         Array containing predicted vegetation index from the model

#     """
        
#     date_as_number=(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-06-23', '%Y-%m-%d')).days
    
#     harmonic_terms = compute_HarmonicTerms(date_as_number)
#     harmonic_terms = xr.DataArray(harmonic_terms, coords={"band" : [1,2,3,4,5]},dims=["band"])
    
#     predicted_vi = sum(coeff_model * harmonic_terms)
#     return predicted_vi


def detection_anomalies(vegetation_index, predicted_vi, threshold_anomaly, vi, path_dict_vi):
    """
    Detects anomalies by comparison between predicted and calculated vegetation index
    
    Parameters
    ----------
    masked_vi : array (x,y) (bool)
        Array, True where pixels are masked
    predicted_vi : array (x,y)
        Array containing predicted vegetation index from the model
    threshold_anomaly : float
        Threshold used to compare predicted and calculated vegetation index. Anomalies are detected if the difference between the two is above this threshold.

    Returns
    -------
    anomalies : array (x,y) (bool)
        Array, pixel value is True if an anomaly is detected.


    """
    dict_vi = get_dict_vi(path_dict_vi)
    
    diff_vi = vegetation_index-predicted_vi
    
    if dict_vi[vi]["decline_change_direction"] == "+":
        anomalies = diff_vi > threshold_anomaly
    elif dict_vi[vi]["decline_change_direction"] == "-":
        anomalies = diff_vi < (-1*threshold_anomaly)
    else:
        raise Exception("Unrecognized decline_change_direction in " + path_dict_vi + " for vegetation index " + vi)
            
    return anomalies

def detection_decline(decline_data, anomalies, mask, date_index):
    """
    Updates decline data using anomalies.
    
    Parameters
    ----------
    decline_data : Dataset with three arrays : 
        "count" which is the number of successive anomalies, 
        "state" which is True where pixels are detected as declining, 
        "first date" which contains the index of the date of the first anomaly.
    anomalies : array (x,y) (bool)
        Array, pixel value is True if an anomaly is detected.
    mask : array (x,y) (bool)
        Array, True where pixels are masked
    date_index : int
        Index of the date

    Returns
    -------
    decline_data : Dataset
        Dataset with the three arrays updated with the results from the date being analysed

    """
   
    decline_data["count"] = xr.where(~mask & (anomalies!=decline_data["state"]),decline_data["count"]+1,decline_data["count"])
    decline_data["count"] = xr.where(~mask & (anomalies==decline_data["state"]),0,decline_data["count"])
    
    decline_data["state"] = xr.where(~mask & (decline_data["count"]==3), ~decline_data["state"], decline_data["state"]) #Changement d'état si CompteurScolyte = 3 et date valide
        
    decline_data["count"] = xr.where(decline_data["count"]==3, 0,decline_data["count"])
    decline_data["first_date"]=xr.where(~mask & (decline_data["count"]==1) & ~decline_data["state"], date_index, decline_data["first_date"]) #Garde la première date de détection de scolyte sauf si déjà détécté comme scolyte
   
    # print(int(decline_data["count"][40,40]))
    # print(int(decline_data["first_date"][40,40]))
    # print(int(decline_data["state"][40,40]))
    
    return decline_data

# def detection_anomalies(masked_vi, predicted_vi, threshold_anomaly):
#     """
#     Détecte les anomalies par comparaison entre l'indice de végétation réel et l'indice prédit.

#     """
        
#     diff_vi = masked_vi["vegetation_index"]-predicted_vi
#     Anomalies = diff_vi > threshold_anomaly
    
#     # Anomalies.where(~(rasterSigma==0),False)#Retire les zones invalides (sans modèle)
    
#     return anomalies