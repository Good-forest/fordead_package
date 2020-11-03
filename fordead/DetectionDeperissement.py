# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:34:34 2020

@author: admin
"""

import datetime
import numpy as np

def PredictVegetationIndex(StackP,date):
    """
    A partir des données du modèle et de la date, prédit l'indice de végétation
    """
    
    NewDay=(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-06-23', '%Y-%m-%d')).days
    PredictedVegetationIndex = StackP[0,:,:] + StackP[1,:,:]*np.sin(2*np.pi*NewDay/365.25) + StackP[2,:,:]*np.cos(2*np.pi*NewDay/365.25) + StackP[3,:,:]*np.sin(2*2*np.pi*NewDay/365.25)+StackP[4,:,:]*np.cos(2*2*np.pi*NewDay/365.25)
    return PredictedVegetationIndex


def DetectAnomalies(VegetationIndex, PredictedVegetationIndex, Mask, rasterSigma, CoeffAnomalie):
    """
    Détecte les anomalies par comparaison entre l'indice de végétation réel et l'indice prédit.

    """
    
    # DiffVegetationIndex = VegetationIndex.where(~Mask)-PredictedVegetationIndex.where(~Mask)
    # Anomalies = DiffVegetationIndex.where(~Mask) > CoeffAnomalie*rasterSigma.where(~Mask)
    
    DiffVegetationIndex = VegetationIndex-PredictedVegetationIndex
    Anomalies = DiffVegetationIndex > CoeffAnomalie*rasterSigma
    
    Anomalies.where(~(rasterSigma==0),False)#Retire les zones invalides (sans modèle)
    
    return Anomalies

