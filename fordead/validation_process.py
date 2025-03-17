# from shapely.geometry import Polygon
# import json
# import rasterio
# import xarray as xr
# import geopandas as gp
# import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

import ast
from scipy.linalg import lstsq
import inspect

from fordead.masking_vi import compute_vegetation_index, get_dict_vi


def filter_args(func,param_dict,combs):
    param_filter_list = func.__code__.co_varnames[:func.__code__.co_argcount]
    args = {variable : dict(zip(param_dict.keys(),combs))[variable] for variable in param_filter_list if variable in dict(zip(param_dict.keys(),combs))}
    return args


def already_existing_test(df1,df2):
    common_columns = df2.columns.intersection(df1.columns)

    all_df = pd.merge(df1[common_columns], df2[common_columns], 
                      on=list(common_columns), how='left', indicator='exists')
        
    already_existing_test_ids = df1[all_df["exists"] == "both"]["test_id"]
    if len(already_existing_test_ids) != 0:
        return already_existing_test_ids.values[0]
    else:
        return None

# def get_test_id(test_info_path, tile_parameters_dataframe):
#     if test_info_path.exists():
#         test_info_dataframe = pd.read_csv(test_info_path, dtype = str)
#         existing_test_id = already_existing_test(test_info_dataframe, tile_parameters_dataframe)
#         if existing_test_id is not None:
#             print("Already existing test with the same parameters (test_id : " + str(existing_test_id) + ")\nResults were not copied\n")
#             test_id =  None
#         else:
#             test_id = test_info_dataframe["test_id"].astype(int).max()+1
#     else:
#         test_id = 1
        
#     return test_id
def custom_converter(value):
    if value == 'None':
        return 'None'
    return value

def get_test_id(test_info_path, tile_parameters_dataframe):
    if test_info_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            test_info_dataframe = pd.read_csv(test_info_path, dtype = str, converters = {'path_dict_vi': custom_converter, 'cloudiness_path': custom_converter})
        # test_info_dataframe = pd.read_csv(test_info_path, converters = {'path_dict_vi': custom_converter})

        existing_test_id = already_existing_test(test_info_dataframe, tile_parameters_dataframe)
        if existing_test_id is not None:
            print("Already existing test with the same parameters (test_id : " + str(existing_test_id) + ")\nResults were not copied\n")
            test_id =  None
        else:
            test_id = test_info_dataframe["test_id"].astype(int).max()+1
    else:
        test_id = 1
        
    return test_id    

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }




    
def combine_validation_results(csv_path_list, merged_csv_path_list, test_info_path,
                                args_dataframe, test_id):
    
    #Di déjà un fichier, comparer les paramètres, si pas de différence : même test, si on retrouve des IdZone, erreur.
    #rajouter csv avec numéros des tests et paramètres
    for i in range(len(csv_path_list)):
        csv_path = csv_path_list[i]
        merged_path = merged_csv_path_list[i]
        data = pd.read_csv(csv_path)
        data["test_id"] = test_id
        data.to_csv(merged_path, mode='a', index=False,header=not(merged_path.exists()))

    args_dataframe["test_id"] = test_id
    args_dataframe.to_csv(test_info_path, mode='a', index=False,header=not(test_info_path.exists()))
    
    



#tested_parameters / tested_parameters_source?
def get_params(params_to_test):    # params_to_test = {}
    
    if isinstance(params_to_test, dict):
        return params_to_test
    #     #Check that variables all exist in process_tile
    elif isinstance(params_to_test, pd.DataFrame):
        param_dict = params_to_test #faux
        return param_dict
    #     #Check that variables (columns names) all exist in process_tile
    elif isinstance(params_to_test, str):
        #Check if all columns or all first rows exist
        #If not in both cases, return problematic variables là où il y en a le moins
        if Path(params_to_test).exists():
            param_dict = {}
            with open(params_to_test, 'r') as f:
                
                for line in f:
                    list_line = line.split()
                    try:
                        param_dict[list_line[0]] = [ast.literal_eval(list_line[1:][i].rstrip('\n')) for i in range(len(list_line[1:]))]
                    except:
                        param_dict[list_line[0]] = list_line[1:]
            return param_dict
        else:
            raise Exception("Text file " + params_to_test + " not found.")
    else:
        raise Exception("Unrecognized params_to_test parameter")
        
        
        
        
        
        
        
# def detect_soil(soil_data, soil_anomaly, invalid, date_index):
#     soil_data["count"]=xr.where(~invalid & soil_anomaly,soil_data["count"]+1,soil_data["count"])
#     soil_data["count"]=xr.where(~invalid & ~soil_anomaly,0,soil_data["count"])
#     soil_data["state"] = xr.where(soil_data["count"] == 3, True, soil_data["state"])
#     soil_data["first_date"] = xr.where(~invalid & (soil_data["count"] == 1) & ~soil_data["state"],date_index,soil_data["first_date"]) #Keeps index of first soil detection
    
#     return soil_data



def get_pre_masks_dataframe(reflect, list_bands, name_column):
    soil_anomaly = compute_vegetation_index(reflect, formula = "(B11 > 1250) & (B2 < 600) & ((B3 + B4) > 800)")
    shadows = (reflect[list_bands]==0).any(axis=1)
    outside_swath = reflect[list_bands[0]]<0
    invalid = shadows | outside_swath | (reflect["B2"] >= 600)
    return soil_anomaly, shadows, outside_swath, invalid

def detect_soil(data_frame, name_column):
    
    data_frame = data_frame[~data_frame["invalid"]].reset_index()
    data_frame["group"] = ((data_frame.soil_anomaly != data_frame.soil_anomaly.shift()).cumsum())
    
    consecutive_soil = data_frame[data_frame["soil_anomaly"]].groupby(by = ["area_name", name_column, "id_pixel", "group"]).size()
    soil_detect_groups = consecutive_soil[consecutive_soil >= 3].reset_index().groupby(["area_name", name_column, "id_pixel"]).first()
    bare_ground_first_date = pd.merge(data_frame, soil_detect_groups, how='inner', on=["area_name", name_column, "id_pixel", "group"]).groupby(["area_name", name_column, "id_pixel"]).first().reset_index()[["area_name",name_column,"id_pixel","Date"]].rename(columns={"Date": "bare_ground_first_date"})
  
    return bare_ground_first_date

def detect_clouds_dataframe(data_frame):
    NG = compute_vegetation_index(data_frame, formula = "B3/(B8A+B4+B3)")
    cond1 = NG > 0.15
    cond2 = data_frame["B2"] > 400
    cond3 = data_frame["B2"] > 700
    cond4 =  ~(data_frame["bare_ground"] | data_frame["soil_anomaly"]) #Not detected as soil
    
    clouds = cond4 & (cond3 | (cond1 & cond2))   
    return clouds

def compute_and_apply_mask(data_frame, soil_detection, formula_mask, list_bands, apply_source_mask, sentinel_source, name_column):
 
    if soil_detection:
        data_frame, bare_ground = compute_masks_dataframe(data_frame, list_bands, name_column)
        # data_frame["bare_ground"] = bare_ground.to_numpy()
    else:
        data_frame = compute_user_mask_dataframe(data_frame, formula_mask, list_bands)
        bare_ground = None
        
    masked_dataframe = data_frame[~data_frame["mask"]]
    
    if apply_source_mask:
        masked_dataframe = masked_dataframe[~source_mask_dataframe(masked_dataframe["Mask"], sentinel_source)]
    
    return masked_dataframe, bare_ground

def compute_user_mask_dataframe(data_frame, formula_mask, list_bands):
    shadows = (data_frame[list_bands]==0).any(axis=1)
    outside_swath = data_frame[list_bands[0]]<0
    user_mask = compute_vegetation_index(data_frame, formula = formula_mask)
    data_frame["mask"] = shadows | outside_swath | user_mask
    return data_frame

def compute_masks_dataframe(data_frame, list_bands, name_column):
    
    data_frame["soil_anomaly"], data_frame["shadows"], data_frame["outside_swath"], data_frame["invalid"] = get_pre_masks_dataframe(data_frame, list_bands, name_column)
    
    bare_ground_first_date = detect_soil(data_frame, name_column)
    
    # soil_data["bare_ground"] = True
    data_frame = data_frame.merge(bare_ground_first_date, on=["area_name", name_column, "id_pixel"], how='left')
 
    data_frame["bare_ground"] = (data_frame["Date"] >= data_frame["bare_ground_first_date"])
    # data_frame["bare_ground"].sum()
    
    # data_frame["bare_ground"] = data_frame.groupby(["area_name", name_column, "id_pixel"]).bare_ground.ffill().fillna(False)

    data_frame["clouds"] = detect_clouds_dataframe(data_frame)
    
    data_frame["mask"] = data_frame["shadows"] | data_frame["clouds"] | data_frame["outside_swath"] | (~data_frame["bare_ground"] & data_frame["soil_anomaly"])
    # mask = shadows | clouds | outside_swath | (data_frame["soil_anomaly"]

    # mask = shadows | clouds | outside_swath | data_frame["bare_ground"] | data_frame["soil_anomaly"]

    
    return data_frame, bare_ground_first_date
  
      
def filter_cloudy_acquisitions(reflect, cloudiness_path, lim_perc_cloud):
    cloudiness = pd.read_csv(cloudiness_path)
    reflect = reflect.merge(cloudiness, how = "left", on = ["area_name", "Date"])
    reflect = reflect[reflect["cloudiness"] <= lim_perc_cloud]
    reflect = reflect.drop(columns=["cloudiness"])
    return reflect
    
def source_mask_dataframe(source_mask, sentinel_source):
    if sentinel_source in ["theia", "theiastac"]:
        source_mask = source_mask>0
    elif sentinel_source in ["scihub", "peps", "Planetary"]:
        source_mask = ~source_mask.isin([4,5])
    return source_mask
        
        
def get_last_training_date_dataframe(data_frame, min_last_date_training, max_last_date_training, nb_min_date=10, name_column = "id"):

    # test = data_frame.groupby(["area_name",name_column,"id_pixel"]).cumcount()+1
    data_frame = data_frame.reset_index()
    data_frame["cumcount"] = (data_frame.groupby(["area_name",name_column,"id_pixel"]).cumcount()+1)
    
    last_training_date = data_frame[(data_frame["Date"] < min_last_date_training) | ((data_frame["cumcount"] <= nb_min_date) & (data_frame["Date"] < max_last_date_training))].groupby(["area_name",name_column,"id_pixel"]).last().reset_index()

    last_training_date.loc[(last_training_date["Date"] >= max_last_date_training) | (last_training_date["cumcount"] < nb_min_date), "Date"] = np.nan
    last_training_date = last_training_date[["epsg","area_name",name_column,"id_pixel","Date"]].rename(columns = {"Date" : "last_training_date"})

    return last_training_date


def compute_HarmonicTerms(DateAsNumber):
    return np.array([1,np.sin(2*np.pi*DateAsNumber/365.25), np.cos(2*np.pi*DateAsNumber/365.25),np.sin(2*2*np.pi*DateAsNumber/365.25),np.cos(2*2*np.pi*DateAsNumber/365.25)])

def model(date_as_number, vi):
    harmonic_terms = np.array([[1]*len(date_as_number), np.sin(2*np.pi*np.array(date_as_number)/365.25), np.cos(2*np.pi*np.array(date_as_number)/365.25), np.sin(2*2*np.pi*np.array(date_as_number)/365.25), np.cos(2*2*np.pi*np.array(date_as_number)/365.25)]).T
    p, _, _, _ = lstsq(harmonic_terms, vi)

    return p

def model_vi_dataframe(data_frame, name_column):
    
    if data_frame.shape[0] == 0:
        raise ValueError("Cannot compute model, data_frame is empty")
    
    data_frame['Date'] = pd.to_datetime(data_frame['Date'])
    data_frame["date_as_number"] = (data_frame['Date'] - pd.to_datetime("2015-01-01")).dt.days
  
    coeff_vi = data_frame.groupby(["area_name",name_column,"id_pixel"]).apply(lambda x: model(x.date_as_number, x.vi), include_groups=False).reset_index()
    
    coeff_vi.columns = ["area_name",name_column,"id_pixel", "coeff"]
    coeff_vi[["coeff1","coeff2","coeff3", "coeff4","coeff5"]] = pd.DataFrame(coeff_vi.coeff.tolist(), index= coeff_vi.index)

    return coeff_vi[["area_name",name_column,"id_pixel", "coeff1","coeff2","coeff3", "coeff4","coeff5"]]

def prediction_vegetation_index_dataframe(masked_vi, pixel_info, name_column):
    
    masked_vi['Date_as_datetime'] = pd.to_datetime(masked_vi['Date'])
    date_as_number = (masked_vi['Date_as_datetime'] - pd.to_datetime("2015-01-01")).dt.days
    
    coeff_model = np.stack(masked_vi["coeff"].values)
    harmonic_terms = np.array([[1]*len(date_as_number), np.sin(2*np.pi*np.array(date_as_number)/365.25), np.cos(2*np.pi*np.array(date_as_number)/365.25), np.sin(2*2*np.pi*np.array(date_as_number)/365.25), np.cos(2*2*np.pi*np.array(date_as_number)/365.25)]).T

    # predicted_vi = prediction_vegetation_index_dataframe(coeff_model,[date])
    
    predicted_vi = np.sum(coeff_model * harmonic_terms,axis = 1)
    
    return predicted_vi

                                
def detection_anomalies_dataframe(data_frame, threshold_anomaly, vi, path_dict_vi):
    
    dict_vi = get_dict_vi(path_dict_vi)
    
    if dict_vi[vi]["dieback_change_direction"] == "+":
        diff_vi = data_frame["vi"] - data_frame["predicted_vi"]
    elif dict_vi[vi]["dieback_change_direction"] == "-":
        diff_vi = data_frame["predicted_vi"] - data_frame["vi"]
    else:
        raise Exception("Unrecognized dieback_change_direction in " + path_dict_vi + " for vegetation index " + vi)
    
    anomalies = diff_vi > threshold_anomaly
    
    return anomalies, diff_vi



    
    
    
def get_stress_index(data_frame, stress_periods):
    # stress_periods = stress_periods.reset_index()
    
    periods = stress_periods[stress_periods.id_pixel == data_frame["id_pixel"].iloc[0]]

    
    stress_index_list = []
    for index, period in periods.iterrows():      
        data_period = data_frame[(data_frame["Date"] >= period['first_date']) & (data_frame["Date"] < period['last_date'])]
        stress_index_list += [(data_period.diff_vi*range(1,len(data_period)+1)).sum()/(len(data_period)*(len(data_period)+1)/2)]

    # id_pixel stress_period_id first_date last_date stress_index
    periods["stress_index"] = stress_index_list
    
    
    return periods.drop(columns=['id_pixel'])
        
    # data_frame[]

def get_state(data):
    
    states = np.where(data["anomaly"], np.where(data["anomaly"], "Stress", "Healthy"), "Healthy")
    if states[-1] == "Stress":
        states[-1] = "Dieback"
    return pd.Series(states)

# df = df.groupby(['category1', 'category2']).apply(myfunc_data)



def detect_state_changes(data_frame, name_column):
    
    data_frame["group"] = ((data_frame.anomaly != data_frame.anomaly.shift()).cumsum()) #Group successive rows with same anomaly value
    consecutive_anomaly = data_frame.groupby(by = ["area_name", name_column,"id_pixel","group", "anomaly"]).size().reset_index(name = "size")     #Count successive rows with same anomaly value
   

    consecutive_anomaly['row_count'] = consecutive_anomaly.groupby(["area_name", name_column,"id_pixel"]).cumcount() + 1

    consecutive_anomaly = consecutive_anomaly[(consecutive_anomaly["size"] >= 3) | (consecutive_anomaly["row_count"] == 1)]
    consecutive_anomaly.loc[(consecutive_anomaly["size"] < 3) & (consecutive_anomaly["row_count"] == 1), "anomaly"] = False

    # test = consecutive_anomaly[(consecutive_anomaly["size"] < 3) & (consecutive_anomaly["row_count"] == 1)]
    # ex = consecutive_anomaly[(consecutive_anomaly.id == 5) & (consecutive_anomaly.id_pixel == 220)]


    consecutive_anomaly["group2"] = ((consecutive_anomaly.anomaly != consecutive_anomaly.anomaly.shift()).cumsum())
    
    # keep first date and get size at the same time could be more efficient
    changes = consecutive_anomaly.groupby(by = ["area_name", name_column,"id_pixel","group2", "anomaly"]).first().reset_index()#.drop(columns=['group2', 'size']) #Remove successive rows with same anomaly value


    #Remove first group of successive false values which is not a change
        # changes = changes[(changes.groupby("id_pixel").cumcount() !=0) | (changes.anomaly)] 
    # changes["period_id"] = changes.groupby(by = ["area_name", name_column,"id_pixel"])['anomaly'].cumsum()
    # changes = changes[changes["period_id"] != 0]
    
    first_dates = data_frame.groupby(by = ["area_name", name_column,"id_pixel","group"]).first().reset_index()[["area_name", name_column, "id_pixel","group","Date"]]
    
    dated_changes = changes.merge(first_dates, on=["area_name", name_column, "id_pixel","group"], how='left') #Rajouter la première date du groupe, perdue pendant le goup_by
    dated_changes["state"] = dated_changes.groupby(["area_name", name_column,"id_pixel"]).apply(get_state, include_groups=False).reset_index().rename(columns = {0:"state"}).state
    dated_changes = dated_changes[['area_name', name_column, 'id_pixel','Date','state']].rename(columns = {"Date" : "first_date"})
    
    
    
    # dated_changes
    # dated_changes["limit"] = np.where(dated_changes['anomaly'], "first_date", "last_date")

    # periods = dated_changes.pivot(index=["area_name", name_column, "id_pixel", "period_id"], columns='limit', values = "Date")
    # periods["state"] = np.where(pd.isnull(periods['last_date']), "Dieback", "Stress")

    return dated_changes

def compute_stress_index(detection_dates, periods, name_column, stress_index_mode):
    
    detection_dates = detection_dates.merge(periods.rename(columns = {"first_date" : "Date"}).drop(columns=["last_date"]), on=["area_name", name_column,"id_pixel","Date"], how='left') 
    detection_dates = detection_dates.ffill()
    
    if stress_index_mode == "mean":
        period_stress_index = detection_dates.groupby(by = ["area_name", name_column,"id_pixel","period_id"]).apply(lambda x : (x.diff_vi.mean()), include_groups=False).reset_index(name = "anomaly_intensity")
    elif stress_index_mode == "weighted_mean":
        period_stress_index = detection_dates.groupby(by = ["area_name", name_column,"id_pixel","period_id"]).apply(lambda x : (x.diff_vi * range(1,len(x)+1)).sum()/(len(x)*(len(x)+1)/2), include_groups=False).reset_index(name = "anomaly_intensity")
    else:
        raise Exception("stress_index_mode '" + stress_index_mode + "' not recognised")
    periods = periods.merge(period_stress_index, on=["area_name", name_column,"id_pixel","period_id"], how='left') 
  
    return periods

def dieback_detection(data_frame, dated_changes, name_column):
    
    dieback = dated_changes[dated_changes["state"] == "dieback"].drop(columns=["group2","anomaly","group","size","state"]).rename(columns = {"Date" : "first_date"})
    # dieback = dated_changes[dated_changes["state"] == "dieback"].rename(columns = {"Date" : "first_date"})[[]]
    dieback_data = data_frame.merge(dieback, on=["area_name", name_column,"id_pixel"], how='left') #Rajouter la première date du groupe, perdue pendant le goup_by
    dieback_data = dieback_data[dieback_data["Date"] >= dieback_data["first_date"]]
    
    dieback["anomaly_intensity"] = dieback_data.groupby(by = ["area_name", name_column,"id_pixel"]).apply(lambda x : (x.diff_vi * range(1,len(x)+1)).sum()/(len(x)*(len(x)+1)/2)).to_numpy()

    return dieback
    # consecutive_anomaly = data_frame.groupby(by = [name_column,"id_pixel","group2", "anomaly"]).size().reset_index(name = "size")

    # consecutive_anomaly[consecutive_anomaly >= 3].groupby(by = [name_column,"id_pixel","group", "anomaly"]).size()
    # anomaly_detect_groups = consecutive_anomaly[consecutive_anomaly >= 3].reset_index().groupby("id_pixel").first()
    # int_df = pd.merge(data_frame, anomaly_detect_groups, how='inner', on=['id_pixel', 'group']).groupby("id_pixel").first().reset_index()[["id_pixel","Date"]]

# def get_training_period(masked_vi, name_column):
    
#     training_dates = masked_vi[masked_vi["Date"] <= masked_vi["last_date_training"]].reset_index()
#     g = training_dates.groupby(by = ["area_name", name_column,"id_pixel"])

#     test = pd.concat([g.head(1), g.tail(1)])
    # masked_vi = masked_vi[masked_vi["Date"] <= masked_vi["last_date_training"]].reset_index()

def get_mask_vi_periods(reflect, first_date_bare_ground, name_column):
    
    
    mask_vi_periods = reflect.groupby(by = ["area_name", name_column,"id_pixel"]).first().reset_index()[["area_name", name_column, "id_pixel", "Date"]].rename(columns = {"Date" : "first_date"})
    mask_vi_periods["state"] = "Training"
    
    if first_date_bare_ground is not None:
        first_date_bare_ground = first_date_bare_ground.rename(columns = {"bare_ground_first_date" : "first_date"})
        first_date_bare_ground["state"] = "Bare ground"
        mask_vi_periods = pd.concat([mask_vi_periods, first_date_bare_ground])
    
    mask_vi_periods = mask_vi_periods.sort_values(["area_name",name_column,"id_pixel"], ascending = True)
    return mask_vi_periods
    # periods_tot.to_csv(export_path, mode='w', index=False,header=True)

def update_training_period(last_date_training, periods_path, name_column):
    last_date_training = last_date_training.rename(columns = {"last_training_date" : "last_date"})
    last_date_training["state"] = "Training"
    
    
    periods = pd.read_csv(periods_path)
    periods = periods.merge(last_date_training, on=["area_name", name_column, "id_pixel","state"], how='left')
    
    periods = periods[["area_name", name_column,"id_pixel","first_date","last_date","state"]]
    
    periods.loc[(periods["state"] == "Training") & (periods["last_date"]).isna(), "state"]  = "Invalid"
    periods.to_csv(periods_path, mode='w', index=False,header=True)


def fill_periods(periods,dated_changes, masked_vi, name_column):
    
    masked_vi["date_before"] = masked_vi.Date.shift()
    masked_vi=masked_vi[["area_name",name_column,"id_pixel","Date","date_before"]].rename(columns = {"Date" : "first_date"})
    
    periods = pd.concat([periods, dated_changes]).sort_values(["area_name",name_column,"id_pixel","first_date"], ascending = True)
   
    #Adding last_date
    periods = periods.merge(masked_vi, on = ["area_name",name_column,"id_pixel","first_date"], how='left')
    periods["date_before_shift"] = periods.date_before.shift(periods = -1)
    
    periods["last_date"] = np.where(periods["state"] != "Training", periods["date_before_shift"], periods["last_date"])
    
    # #Adding first healthy period without anomalies
    # masked_vi = masked_vi.rename(columns = {"first_date" : "last_date"})
    # masked_vi["date_after"] = masked_vi.last_date.shift(periods = -1)
    # test = test.merge(masked_vi, on = ["area_name",name_column,"id_pixel","last_date"], how='left')
    periods['period_id'] = periods.groupby(["area_name", name_column,"id_pixel"]).cumcount()
    periods = periods[["area_name",name_column,"id_pixel","period_id","state","first_date","last_date"]]
    return periods
    
    # check = test[~test.last_date.isna() & (test.date_before_shift != test.last_date)]
    # ex = test[(test.id == 5) & (test.id_pixel == 220)]
    
def add_diff_vi_to_vi(masked_vi, pixel_info, threshold_anomaly, vi, path_dict_vi, name_column):
    masked_vi = masked_vi.merge(pixel_info, on=["epsg", "area_name",name_column,"id_pixel"], how='left')
    valid_vi = masked_vi[~masked_vi["last_training_date"].isna()].reset_index()
    valid_vi["predicted_vi"] = prediction_vegetation_index_dataframe(valid_vi, pixel_info, name_column)
    valid_vi["anomaly"], valid_vi["diff_vi"] = detection_anomalies_dataframe(valid_vi, threshold_anomaly, vi = vi, path_dict_vi = path_dict_vi)
    
    valid_vi = valid_vi[['epsg', 'area_name', name_column, 'id_pixel', 'Date','predicted_vi', 'diff_vi','anomaly']]
    masked_vi = masked_vi[['epsg', 'area_name', name_column, 'id_pixel', 'Date', 'vi']]
    masked_vi = masked_vi.merge(valid_vi, on=["epsg", "area_name",name_column,"id_pixel","Date"], how='left')
    # masked_vi.loc[~masked_vi["last_training_date"].isna(),["predicted_vi","anomaly","diff_vi"]] = valid_vi.loc[:,["predicted_vi","anomaly","diff_vi"]]

    return masked_vi

def add_status_to_vi(masked_vi, periods, name_column, stress_index_mode):
        
    if stress_index_mode is not None:
        masked_vi = masked_vi.merge(periods.rename(columns = {"first_date" : "Date"}).drop(columns=["last_date","anomaly_intensity"]), on=["area_name", name_column,"id_pixel","Date"], how='left') 
    else:
        masked_vi = masked_vi.merge(periods.rename(columns = {"first_date" : "Date"}).drop(columns=["last_date"]), on=["area_name", name_column,"id_pixel","Date"], how='left') 

    if "bare_ground" in masked_vi.columns:
        masked_vi = masked_vi.drop(columns=["bare_ground"])
    for col in ['period_id', 'state']:
        masked_vi[col] = masked_vi[col].ffill()

    return masked_vi
    # period_stress_index = masked_vi.groupby(by = ["area_name", name_column,"id_pixel","period_id"]).apply(lambda x : (x.diff_vi * range(1,len(x)+1)).sum()/(len(x)*(len(x)+1)/2)).reset_index(name = "anomaly_intensity")
    # periods = periods.merge(period_stress_index, on=["area_name", name_column,"id_pixel","period_id"], how='left') 


################################################
#tested_parameters / tested_parameters_source?
def get_args(params_to_test):    # params_to_test = {}
    
    if isinstance(params_to_test, dict):
        return params_to_test
    #     #Check that variables all exist in functions
    # elif isinstance(params_to_test, pd.DataFrame):
    #     param_dict = params_to_test #faux
    #     return param_dict
    #     #Check that variables (columns names) all exist in functions
    elif isinstance(params_to_test, str):
        #Check if all columns or all first rows exist
        #If not in both cases, return problematic variables là où il y en a le moins
        if Path(params_to_test).exists():
            param_dict = {}
            with open(params_to_test, 'r') as f:
                
                for line in f:
                    list_line = line.split()
                    try:
                        param_dict[list_line[0]] = [ast.literal_eval(list_line[1:][i].rstrip('\n')) for i in range(len(list_line[1:]))]
                    except:
                        param_dict[list_line[0]] = list_line[1:]
            return param_dict
        else:
            raise Exception("Text file " + params_to_test + " not found.")
    else:
        raise Exception("Unrecognized params_to_test parameter")
        

# def combine_validation_results(data_directory, target_directory):
    
#     #Di déjà un fichier, comparer les paramètres, si pas de différence : même test, si on retrouve des IdZone, erreur.
#     #rajouter csv avec numéros des tests et paramètres
    
#     tile = TileInfo(data_directory)
#     tile = tile.import_info()

#     state_obs_path = target_directory / 'state_obs.csv'
#     pixel_obs_path = target_directory / 'pixel_obs_info.csv'
#     test_info_path = target_directory / 'test_info.csv'
    
#     tile.add_dirpath("compared_validation", target_directory)

#     state_dataframe = pd.read_csv(tile.paths["validation"] / 'state_obs.csv')
#     pixel_info_dataframe = pd.read_csv(tile.paths["validation"] / 'pixel_obs_info.csv')
        
#     tile_parameters = tile.parameters
#     for key in tile_parameters:
#         tile_parameters[key] = str(tile_parameters[key])
#     del tile.parameters["Overwrite"] #Overwrite ne devrait pas être dans les paramètres
#     tile_parameters_dataframe = pd.DataFrame(data=tile_parameters, index=[0])
    
#     test_id = get_test_id(test_info_path, tile_parameters_dataframe)
    
#     if test_id is not None:
#         tile_parameters_dataframe["test_id"] = test_id
#         state_dataframe["test_id"] = test_id
#         pixel_info_dataframe["test_id"] = test_id

#         tile_parameters_dataframe.to_csv(test_info_path, mode='a', index=False,header=not(test_info_path.exists()))
    
            
#         state_dataframe.to_csv(state_obs_path, mode='a', index=False,header=not(state_obs_path.exists()))
#         pixel_info_dataframe.to_csv(pixel_obs_path, mode='a', index=False,header=not(pixel_obs_path.exists()))

# def rasterize_obs(obs_shape, name_column, raster_metadata):
#     obs_shape_json_str = obs_shape.to_json()
#     obs_shape_json_dict = json.loads(obs_shape_json_str)
#     obs_shape_jsonMask = [(feature["geometry"],feature["properties"][name_column]) for feature in obs_shape_json_dict["features"]]
    
#     rasterized_validation_data = rasterio.features.rasterize(obs_shape_jsonMask,out_shape = (raster_metadata["sizes"]["y"],raster_metadata["sizes"]["x"]) ,dtype="int16",transform=raster_metadata['transform'])  

#     return xr.DataArray(rasterized_validation_data,coords={"y" : raster_metadata["coords"]["y"],"x" : raster_metadata["coords"]["x"]}, dims=["y","x"])


# def attribute_area_to_obs(obs_shape, areas, name_column):

#     obs_area_tot = obs_shape[[name_column]]
#     obs_area_tot.insert(1, "area_tot", obs_shape.area)
    
#     obs_shape = obs_shape.merge(obs_area_tot, on= name_column, how='left')
#     obs_intersection = gp.overlay(obs_shape, areas)
    
#     obs_intersection.insert(1, "area_intersect", obs_intersection.area)
#     obs_intersection = obs_intersection[obs_intersection["area_tot"].round(3) == obs_intersection['area_intersect'].round(3)]
    
#     unvalid_obs = obs_shape[~obs_shape[name_column].isin(obs_intersection[name_column])]
#     if len(unvalid_obs) != 0:
#         print("Observations not fitting entirely on one tile found.\nThey are removed from the dataset. \nIds are the following : \n" + ', '.join(map(str, unvalid_obs[name_column])) + " ")
    
#     #Get observations with only one associated tile
#     count_data = pd.DataFrame(obs_intersection[name_column].value_counts().reset_index())
#     count_data.columns = [name_column, 'count']
#     merged = pd.merge(obs_intersection, count_data, on= name_column)
#     one_tile_obs = merged[merged["count"] == 1]
#     one_tile_obs = one_tile_obs[[name_column,"area_name"]]
    
    
#     processed_dataset =  obs_shape[obs_shape[name_column].isin(one_tile_obs[name_column])]
#     processed_dataset = processed_dataset.merge(one_tile_obs, on= name_column, how='left')
    
#     #Get observations with choice in the tile, attributing the most frequent tile
#     count_data2 = pd.DataFrame(processed_dataset["area_name"].value_counts().reset_index())
#     count_data2.columns = ["area_name", 'count']

#     unattributed_obs = obs_intersection[~obs_intersection[name_column].isin(processed_dataset[name_column])]
#     unattributed_obs = unattributed_obs.merge(count_data2, on= "area_name", how='left')
#     unattributed_obs = unattributed_obs[~pd.isnull(unattributed_obs["count"])]
    
#     choice = unattributed_obs.sort_values([name_column,'count'], ascending = False).groupby(name_column, as_index=False).first()[[name_column,"area_name"]]
#     obs_choice =  obs_shape[obs_shape[name_column].isin(choice[name_column])]
#     obs_choice = obs_choice.merge(choice, on= name_column, how='left')
    
#     processed_dataset = pd.concat([processed_dataset,obs_choice])
    
    
#     #Retrieve those with no associated tiles.
#     unattributed_obs = obs_intersection[~obs_intersection[name_column].isin(processed_dataset[name_column])]
#     while len(unattributed_obs)!=0:
#         count_data3 = pd.DataFrame(unattributed_obs["area_name"].value_counts().reset_index())
#         count_data3.columns = ["area_name", 'count']
#         choice = obs_intersection[obs_intersection["area_name"]==count_data3.sort_values(['count'], ascending = False).area_name[0]][[name_column,"area_name"]]
#         obs_choice =  obs_shape[obs_shape[name_column].isin(choice[name_column])]
#         obs_choice = obs_choice.merge(choice, on= name_column, how='left')
#         processed_dataset = pd.concat([processed_dataset,obs_choice])
#         unattributed_obs = obs_intersection[~obs_intersection[name_column].isin(processed_dataset[name_column])]
#     #######
#     processed_dataset = processed_dataset.drop(columns=["area_tot"])
#     return processed_dataset

# def compute_stress_index(data_frame, periods, name_column):
    
#     period_id = 1
#     period_exists = True
#     period_data_list = []
#     period_data_list += [periods[periods["period_id"] == 0]]
#     while period_exists:
#         print(period_id)
#         # stress_period = periods.groupby(["area_name", name_column,'id_pixel']).nth(period_id)
#         period = periods[periods["period_id"] == period_id]
#         # period = period[periods["state"].isin(["Stress","Dieback"])]
        
#         # if period_id == 12:
#         #     print("test")
#         if len(period) != 0 :
#             # data_frame = data_frame.reset_index()
#             # period = period.reset_index()
#             filtered_period_dataframe = data_frame.merge(period, on=["area_name", name_column,"id_pixel"], how='left') 
#             # filtered_period_dataframe = filtered_period_dataframe[(filtered_period_dataframe["Date"] >= filtered_period_dataframe["first_date"]) & (pd.isnull(filtered_period_dataframe["last_date"]) | (filtered_period_dataframe["Date"] < filtered_period_dataframe["last_date"]))]
#             filtered_period_dataframe = filtered_period_dataframe[(filtered_period_dataframe["Date"] >= filtered_period_dataframe["first_date"]) & (filtered_period_dataframe["Date"] <= filtered_period_dataframe["last_date"])]
#             period = period.set_index(["area_name", name_column,"id_pixel", "first_date"])
#             period["anomaly_intensity"] = filtered_period_dataframe.groupby(by = ["area_name", name_column,"id_pixel","first_date"]).apply(lambda x : (x.diff_vi * range(1,len(x)+1)).sum()/(len(x)*(len(x)+1)/2))
#             # test = filtered_period_dataframe.groupby(by = ["area_name", name_column,"id_pixel","first_date"]).apply(lambda x : (x.diff_vi * range(1,len(x)+1)).sum()/(len(x)*(len(x)+1)/2))


#             # stress_period["period_id"] = i+1
            
#             period_data_list += [period.reset_index()]
#         else:
#             period_exists = False

#         period_id+=1
        
#     period_data = pd.concat(period_data_list)
    
#     period_data = period_data[["area_name", name_column,"id_pixel","state","first_date","last_date", "anomaly_intensity"]].sort_values(["area_name",name_column,"id_pixel","first_date"], ascending = True)

#     return period_data


# def get_conf_index(data_frame, dieback):
#     periods = dieback[dieback.id_pixel == data_frame["id_pixel"].iloc[0]]

    
#     stress_index_list = []
#     for index, period in periods.iterrows():      
#         data_period = data_frame[(data_frame["Date"] >= period['first_date']) & (data_frame["Date"] < period['last_date'])]
#         stress_index_list += [(data_period.diff_vi*range(1,len(data_period)+1)).sum()/(len(data_period)*(len(data_period)+1)/2)]

#     # id_pixel stress_period_id first_date last_date stress_index
#     periods["stress_index"] = stress_index_list
    
    
#     return periods.drop(columns=['id_pixel'])


