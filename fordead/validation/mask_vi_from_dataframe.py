# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:48:06 2023

@author: Raphaël Dutrieux
"""

from fordead import __version__
import click
import pandas as pd
from fordead.validation_process import compute_and_apply_mask, filter_cloudy_acquisitions, get_mask_vi_periods
from fordead.masking_vi import compute_vegetation_index
from fordead.cli.utils import empty_to_none
from pathlib import Path
import time
import numpy as np
#Pas besoin de recalculer les masques si compute_vegetation_index change
#Sortir première date de bare_ground

@click.command(name='calval_masked_vi')
@click.option("--reflectance_path", type = str, help = "Path of the csv file with extracted reflectance.")
@click.option("--masked_vi_path", type = str, help = "Path used to write the csv containing the vegetation index for each pixel of each observation, for each valid Sentinel-2 acquisition.")
@click.option("--periods_path", type = str, help = "Path used to write the csv containing the first date of the training periods for each pixel and, if soil_detection is True, the first date of detected bare ground.")
@click.option("--name_column", type = str,default = "id", help = "Name of the ID column", show_default=True)
@click.option("--cloudiness_path", type = str, default = None, help = "Path of a csv with the columns 'area_name','Date' and 'cloudiness' used to filter acquisitions, can be calculated by the [extract_cloudiness function](https://fordead.gitlab.io/fordead_package/docs/Tutorials/Validation/03_extract_cloudiness/). Not used if not given.", show_default=True)
@click.option("--vi", type = str,default = "CRSWIR", help = "Chosen vegetation index", show_default=True)
@click.option("-n", "--lim_perc_cloud", type = float,default = 0.4, help = "Maximum cloudiness at the tile scale, used to filter used SENTINEL dates. Set parameter as -1 to not filter based on cloudiness", show_default=True)
@click.option("--soil_detection",  is_flag=True, help = "If True, bare ground is detected and used as mask,"
                                                        " but the process has not been tested on other data "
                                                        "than THEIA data in France (see https://fordead.gitlab.io/fordead_package/docs/user_guides/english/01_compute_masked_vegetationindex/)."
                                                        " If False, mask from formula_mask is applied.", show_default=True)
@click.option("--formula_mask", type = str,default = "(B2 >= 700)", help = "formula whose result would be binary, as described here https://fordead.gitlab.io/fordead_package/reference/fordead/masking_vi/#compute_vegetation_index. Is only used if soil_detection is False.", show_default=True)
@click.option("--path_dict_vi", type = str,default = None, help = "Path of text file to add vegetation index formula, if None, only built-in vegetation indices can be used (CRSWIR, NDVI)", show_default=True)
@click.option("-b","--list_bands", type=str, multiple=True, default = ["B2","B3","B4", "B8", "B8A", "B11","B12"], help = "Bands to import and use ex : -b B2 -b  B3 -b B11", show_default=True) # ["B2","B3","B4","B5","B6","B7","B8","B8A","B11", "B12", "Mask"]
@click.option("--apply_source_mask",  is_flag=True, help = "If True, applies the mask from SENTINEL-data supplier", show_default=True)
@click.option("--sentinel_source", type=str, default = "theia", help = "Source of data, can be 'theia' et 'scihub' et 'peps'", show_default=True)
@click.option("--ignored_period", multiple=True, type = str, default = None, help = "Period whose Sentinel dates to ignore (format 'MM-DD', ex : --ignored_period 11-01 --ignored_period 05-01", show_default=True)
def cli_mask_vi_from_dataframe(**kwargs):
    """
    Computes the vegetation index for each pixel of each observation, for each valid Sentinel-2 acquisition.
    Filters out data by applying the fordead mask (if soil_detection is True) or a user mask defined by the user. 
    (optional) Filters out acquisition by applying a limit on the percentage of cloud cover as calculated by the [extract_cloudiness function](https://fordead.gitlab.io/fordead_package/docs/Tutorials/Validation/03_extract_cloudiness/)
    Writes the results in a csv file, as well as the first date of the training period for each pixel and, if soil_detection is True, the first date of detected bare ground.
    
    See additional information [here](https://fordead.gitlab.io/fordead_package/docs/user_guides/english/validation_tools/05_compute_masks_and_vegetation_index_from_dataframe/)
    """
    # Remove args with multiple, as default is an empty tuple
    empty_to_none(kwargs, "list_bands")
    empty_to_none(kwargs, "ignored_period")
    print(kwargs)
    mask_vi_from_dataframe(**kwargs)


def mask_vi_from_dataframe(reflectance_path,
                           masked_vi_path,
                           periods_path,
                           name_column,
                           cloudiness_path = None,
                           vi = "CRSWIR",
                           lim_perc_cloud = 0.4,
                           soil_detection = True,
                           formula_mask = "(B2 >= 700)",
                           path_dict_vi = None,
                           list_bands =  ["B2","B3","B4", "B8", "B8A", "B11","B12"],
                           apply_source_mask = False,
                           sentinel_source = "theia",
                           ignored_period = None
                           ):
    """
    Computes the vegetation index for each pixel of each observation, for each valid Sentinel-2 acquisition.
    Filters out data by applying the fordead mask (if soil_detection is True) or a user mask defined by the user. 
    (optional) Filters out acquisition by applying a limit on the percentage of cloud cover as calculated by the [extract_cloudiness function](https://fordead.gitlab.io/fordead_package/docs/Tutorials/Validation/03_extract_cloudiness/)
    Writes the results in a csv file, as well as the first date of the training period for each pixel and, if soil_detection is True, the first date of detected bare ground.
    
    See additional information [here](https://fordead.gitlab.io/fordead_package/docs/user_guides/english/validation_tools/05_compute_masks_and_vegetation_index_from_dataframe/)
    
    Parameters
    ----------
    reflectance_path : str
        Path of the csv file with extracted reflectance.
    masked_vi_path : str
        Path used to write the csv containing the vegetation index for each pixel of each observation, for each valid Sentinel-2 acquisition.
    periods_path : str
        Path used to write the csv containing the first date of the training periods for each pixel and, if soil_detection is True, the first date of detected bare ground.
    name_column : str
        Name of the ID column. The default is 'id'.
    cloudiness_path : str (optional)
        Path of a csv with the columns 'area_name','Date' and 'cloudiness' used to filter acquisitions, can be calculated by the [extract_cloudiness function](https://fordead.gitlab.io/fordead_package/docs/Tutorials/Validation/03_extract_cloudiness/). Not used if None.
    vi : str, optional
        Chosen vegetation index. If using a custom one or one unavailable in this package, it can be added using the path_dict_vi parameter. The default is "CRSWIR".
    lim_perc_cloud : float, optional
        The maximum percentage of clouds of the whole area. If the cloudiness percentage of the SENTINEL acquisition, calculated from the provider's classification, is higher than this threshold, the acquisition is filtered out. Only used if cloudiness_path is not None. The default is 0.45.
    soil_detection : bool, optional
        If True, bare ground is detected and used as mask, but the process might not be adapted to other situations than THEIA data on France's coniferous forests. If False, mask from formula_mask is applied. The default is True.
    formula_mask : str, optional
        Formula whose result would be binary, format described [here](https://fordead.gitlab.io/fordead_package/reference/fordead/masking_vi/#compute_vegetation_index). Is only used if soil_detection is False.. The default is "(B2 >= 700)".
    path_dict_vi : str, optional
        Path to a text file used to add potential vegetation indices. If not filled in, only the indices provided in the package can be used (CRSWIR, NDVI, NDWI). The file [ex_dict_vi.txt](https://gitlab.com/fordead/fordead_package/-/blob/master/docs/examples/ex_dict_vi.txt) gives an example for how to format this file. One must fill the index's name, formula, and "+" or "-" according to whether the index increases or decreases when anomalies occur.. The default is None.
    list_bands : list of str, optional
        List of the bands used (do not include mask as 0 are considered as NaN and removed).
        The default is ["B2","B3","B4", "B8", "B8A", "B11","B12"].
    apply_source_mask : bool, optional
        If True, the mask of the provider is also used to mask the data. The default is False.
    sentinel_source : str, optional
        Provider of the data among 'theia' and 'scihub' and 'peps'.. The default is "theia".
    ignored_period : list of two strings
        Period whose Sentinel acquisitions to ignore (format 'MM-DD', ex : ["11-01","05-01"])
    """
    

    reflect = pd.read_csv(reflectance_path)
    
    if cloudiness_path is not None:
        reflect = filter_cloudy_acquisitions(reflect, cloudiness_path, lim_perc_cloud)
        
    if ignored_period is not None:
        reflect['Date'] = reflect['Date'].astype(str)
        reflect = reflect[(reflect["Date"].str[5:] > min(ignored_period)) & (reflect["Date"][5:].str[5:] < max(ignored_period))]        
    
    reflect = reflect.sort_values(by=["area_name",name_column,'id_pixel', 'Date'])
    
    reflect, first_date_bare_ground = compute_and_apply_mask(reflect, soil_detection, formula_mask, list_bands, apply_source_mask, sentinel_source, name_column)

    reflect["vi"] = compute_vegetation_index(reflect, vi = vi, path_dict_vi = path_dict_vi)
    reflect = reflect[~reflect["vi"].isnull()]
    reflect = reflect[~np.isinf(reflect["vi"])]
    
    if soil_detection:
        reflect = reflect[["epsg", "area_name", name_column, "id_pixel", "Date","vi", "bare_ground"]]
    else:
        reflect = reflect[["epsg", "area_name", name_column, "id_pixel", "Date","vi"]]
    # reflect = reflect.drop(columns=list_bands + ["soil_anomaly", "Mask"]) #soil_anomaly shouldn't be added in the first place
    
    mask_vi_periods = get_mask_vi_periods(reflect, first_date_bare_ground, name_column)
    
    mask_vi_periods.to_csv(periods_path, mode='w', index=False,header=True)
    reflect.to_csv(masked_vi_path, mode='w', index=False,header=True)


if __name__ == '__main__':
    start_time_debut = time.time()
    
    # Exemple tuto
    mask_vi_from_dataframe(reflectance_path = "D:/fordead/fordead_data/output/reflectance_tuto.csv",
                        masked_vi_path = "D:/fordead/fordead_data/output/calval_tuto/mask_vi_tuto.csv",
                        periods_path = "D:/fordead/fordead_data/output/calval_tuto/periods_tuto.csv",
                        cloudiness_path = "D:/fordead/fordead_data/output/cloudiness_tuto.csv",
                        vi = "CRSWIR",
                        name_column = "id")


    print("Temps de calcul : %s secondes ---" % (time.time() - start_time_debut))
