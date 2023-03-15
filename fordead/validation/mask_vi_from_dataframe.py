# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:48:06 2023

@author: Raphaël Dutrieux
"""

import pandas as pd
from fordead.validation_process import compute_and_apply_mask
from fordead.masking_vi import compute_vegetation_index
from pathlib import Path
import time
import numpy as np
#Pas besoin de recalculer les masques si compute_vegetation_index change
#Sortir première date de bare_ground


def compute_masked_vi(reflectance_path,
                           export_path,
                           name_column,
                           vi,
                           soil_detection = True,
                           formula_mask = "(B2 >= 700)",
                           path_dict_vi = None,
                           list_bands =  ["B2","B3","B4","B5", "B6", "B7", "B8", "B8A", "B11","B12"],
                           apply_source_mask = False,
                           sentinel_source = "THEIA"
                           ):
    
    reflect = pd.read_csv(reflectance_path)
    reflect = reflect.sort_values(by=["area_name",name_column,'id_pixel', 'Date'])
    
    reflect = compute_and_apply_mask(reflect, soil_detection, formula_mask, list_bands, apply_source_mask, sentinel_source, name_column)

    reflect["vi"] = compute_vegetation_index(reflect, vi = "CRSWIR", formula = None, path_dict_vi = None)
    reflect = reflect[~reflect["vi"].isnull()]
    reflect = reflect[~np.isinf(reflect["vi"])]
    
    reflect = reflect.drop(columns=list_bands + ["soil_anomaly", "Mask"]) #soil_anomaly shouldn't be added in the first place
    reflect.to_csv(export_path, mode='w', index=False,header=True)


if __name__ == '__main__':
    start_time_debut = time.time()
    
    # compute_masked_vi(reflectance_path = "D:/fordead/fordead_data/output/reflectance_tuto.csv",
    #                    export_path = "D:/fordead/fordead_data/output/mask_vi_tuto.csv",
    #                    vi = "CRSWIR",
    #                    name_column = "id")
    
    compute_masked_vi(reflectance_path = "D:/fordead/fordead_data/output/reflectance_tuto.csv",
                       export_path = "D:/fordead/fordead_data/output/mask_vi_tuto.csv",
                       vi = "CRSWIR",
                       name_column = "id")
    print("Temps de calcul : %s secondes ---" % (time.time() - start_time_debut))
