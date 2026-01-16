# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:21:35 2020

@author: Raphael Dutrieux
"""
import xarray as xr
import numpy as np
import dask.array as da
import datetime
from scipy.linalg import lstsq
from fordead.import_data import import_binary_raster, import_masked_vi
import warnings

from pyproj import Transformer

def get_detection_dates(stack_masks,min_last_date_training,nb_min_date=10):
    """
    Returns the index of the last date which will be used for training from the masks.
    
    Parameters
    ----------
    stack_masks : xarray.DataArray (Time,x,y)
        Stack of masks with dimensions (Time,x,y)
    min_last_date_training : str
        Earliest date at which the training ends and the detection begins
    nb_min_date : int, optional
        Minimum number of dates used to train the model. The default is 10.

    Returns
    -------
    detection_dates : xarray.DataArray (Time,x,y)
        Boolean array, True at dates where the pixel is used for detection, False when used for training
    first_detection_date_index : xarray.DataArray (x,y)
        Array containing the index of the last date which will be used for training, or 0 if there isn't enough valid data.
    """
    
    min_date_index=int(sum(stack_masks.Time<min_last_date_training))-1
    
    if min_date_index == stack_masks.Time.size-1 :
        warnings.warn(f"Changing min_last_date_training to {stack_masks.Time.values[min_date_index-1]} to have enough data for detection. Extend max_last_date_training to avoid that warning.")
        min_date_index = min_date_index-1
    indexes = xr.DataArray(da.ones(stack_masks.shape,dtype=np.uint16, chunks=stack_masks.chunks), stack_masks.coords) * xr.DataArray(range(stack_masks.sizes["Time"]), coords={"Time" : stack_masks.Time},dims=["Time"])   
    cumsum=(~stack_masks).cumsum(dim="Time",dtype=np.uint16)
    detection_dates = ((indexes > min_date_index) & (cumsum > nb_min_date))
    first_detection_date_index = detection_dates.argmax(dim="Time").astype(np.uint16)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    lat, lon = 48.00444378, 6.79535380
    utm_x, utm_y = transformer.transform(lon, lat)
    pixel_mask = stack_masks.sel(
        x=utm_x,
        y=utm_y,
        method="nearest"
    ).compute()
    valid_times = pixel_mask.Time.where(pixel_mask, drop=True)
    unix_ts = valid_times.values

    np.savetxt("/tmp/timestamps.txt", unix_ts, fmt='%s')

    return detection_dates, first_detection_date_index


def compute_HarmonicTerms(DateAsNumber):
    return np.array([1,np.sin(2*np.pi*DateAsNumber/365.25), np.cos(2*np.pi*DateAsNumber/365.25),np.sin(2*2*np.pi*DateAsNumber/365.25),np.cos(2*2*np.pi*DateAsNumber/365.25)])


def model_vi(stack_vi, stack_masks, one_dim = False):      

    """
    Models periodic vegetation index for each pixel.  

    Parameters
    ----------
    stack_vi : array (Time,x,y)
        Array containing vegetation index data, must contain coordinates 'Time' with format '%Y-%m-%d'
    stack_masks : array (Time,x,y)
        Array (boolean) containing mask data.
    one_dim : bool
        Has to be True if data dimension is only temporal, and have no spatial dimensions

    Returns
    -------
    coeff_model : array (5,x,y)
        Array containing the five coefficients of the vegetation index model for each pixel

    """
    
    DatesNumbers = [(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days for date in np.array(stack_vi["Time"])]
    
    HarmonicTerms = np.array([compute_HarmonicTerms(DateAsNumber) for DateAsNumber in DatesNumbers])
    if not one_dim:
        coeff_model = xr.map_blocks(censored_lstsq, stack_vi, args=[~stack_masks], kwargs={'A':HarmonicTerms}).rio.write_crs(stack_vi.rio.crs)
        coeff_model['coeff'] = range(1,6) # coordinate values as recorded in .tif bands
        
    else:
        p, _, _, _ = lstsq(HarmonicTerms[~stack_masks][0], stack_vi.where(~stack_masks,drop=True))
        coeff_model = xr.DataArray(p, coords={"coeff" : range(1,6)},dims=["coeff"])
        
    return coeff_model

def censored_lstsq(B, M, A):
    """Solves least squares problem subject to missing data, compatible with numpy, dask and xarray.

    Code inspired from http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq

    Parameters
    ----------
    B : (N, ...) numpy.ndarray or xarray.DataArray
    M : (N, ...) boolean numpy.ndarray or xarray.DataArray
        Mask giving the missing data (when False the data is not used in computation).
        It must have the same size/chunks as B.
    A : (N, P) numpy.array


    Returns
    -------
    X : (P, ...) numpy.ndarray that minimizes norm(M*(AX - B))

    Notes
    -----
    If `B` is two-dimensional (e.g. (N, K)), the least-squares solution is calculated for each of column
    of `B`, returning X with dimension (P, K)

    If B is more than 2D, the computation is made with a reduced to 2D keeping the first dimension to its original size. Same for M.

    It should be used with xr.map_blocks (only full A is needed, B et M can be processed by blocks).

    It uses a broadcasted solve for speed.

        Examples
    --------

    # with numpy array

    # with xarray
    import xarray as xr
    import numpy as np
    xysize = 1098
    tsize = 10
    chunksxy = 100
    chunks3D = {'x':chunksxy, 'y':chunksxy, 'Time':10}
    chunks2D = {'x':chunksxy, 'y':chunksxy}

    B = xr.DataArray(np.arange(np.prod([tsize,xysize,xysize])).reshape((tsize,xysize,xysize)), dims=('Time', "x", "y"),
                            coords=[('Time', np.arange(tsize)), ('x', np.arange(xysize)), ('y', np.arange(xysize))]).chunk(chunks=chunks3D)

    M = xr.DataArray(np.random.rand(tsize,xysize,xysize), dims=('Time', "x", "y"),
                            coords=[('Time', np.arange(tsize)), ('x', np.arange(xysize)), ('y', np.arange(xysize))]).chunk(chunks=chunks3D)>0.05

    A = np.random.rand(tsize, 5)

    # The following are 3 different ways for the same result
    ## xarray
    res = xr.map_blocks(censored_lstsq, B, args=[M], kwargs={'A':A})

    ## dask array blockwise
    res = da.array.blockwise(censored_lstsq, 'kmn', B.data, 'tmn', M.data, 'tmn', new_axes={'k':5}, dtype=A.dtype, A=A, meta=np.ndarray(()))
    res = xr.DataArray(res, dims=['coeff', B.dims[1], B.dims[2]],
                       coords=[('coeff', np.arange(A.shape[1])), B.coords[B.dims[1]],
                               B.coords[B.dims[2]]])

    ## dask map_blockss
    res = da.array.map_blocks(censored_lstsq, B.data, M.data, A=A,
                              chunks=((A.shape[1],), B.chunks[1], B.chunks[2]),
                              meta=np.ndarray(()), dtype=np.float_)
    res = xr.DataArray(res, dims=['coeff', B.dims[1], B.dims[2]],
                       coords=[('coeff', np.arange(A.shape[1])), B.coords[B.dims[1]],
                               B.coords[B.dims[2]]])

    """

    # compatibility with xarray
    if isinstance(B, xr.DataArray):
        if np.sum(B.shape) == 0:  # helps xr.map_blocks finding the template on its own
            res = np.empty((A.shape[1], 0, 0))
        else:
            res = censored_lstsq(B.data, M.data, A)

        out = xr.DataArray(res, dims=['coeff', B.dims[1], B.dims[2]],
                           coords=[('coeff', np.arange(res.shape[0])), B.coords[B.dims[1]],
                                   B.coords[B.dims[2]]])
        return out

    # compatibility with dask.array.blockwise
    if isinstance(B, list):
        B = B[0]
        M = M[0]

    # Note: we should check A is full rank but we won't bother...
    shape = B.shape
    B = B.reshape((shape[0], -1))
    M = M.reshape(shape[0], -1)

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[M], B[M])[0]

    out = np.empty((A.shape[1], M.shape[1]), dtype=A.dtype)
    out[:] = np.nan
    valid_index = (M.sum(axis=0) > A.shape[1])

    B = B[:, valid_index]
    M = M[:, valid_index]
    
    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor
    out[:, valid_index] = np.squeeze(np.linalg.solve(T, rhs)).T  # transpose to get r x n
    out = out.reshape([A.shape[1]] + list(shape[1:]))
    # del B, M, rhs, T
    return out

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
        
    date_as_number_list=[(datetime.datetime.strptime(date, '%Y-%m-%d')-datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')).days for date in date_list]
    harmonic_terms = np.array([compute_HarmonicTerms(DateAsNumber) for DateAsNumber in date_as_number_list])
    harmonic_terms = xr.DataArray(harmonic_terms, coords={"Time" : date_list, "coeff" : range(1, 6)},dims=["Time", "coeff"])
    
    predicted_vi = sum(coeff_model * harmonic_terms)
    
    return predicted_vi

def model_vi_correction(stack_vi, stack_masks, dict_paths):
    """
    Corrects stacked vegetation index using large scale vegetation index median value.
    A periodic model is fitted on the median vegetation index of unmasked pixels within the forest mask. The difference between the prediction of this model and these large scale median values is used as a correction term to correct the vegetation index.
    This is meant to take into account changes affecting all pixels equally which are not linked with a dieback.

    Parameters
    ----------
    stack_vi : array (Time,x,y)
        array containing uncorrected vegetation index data
    stack_masks : array (Time,x,y)
        Array (boolean) containing mask data.
    dict_paths : dict
        Dictionnary containing vegetation index path for each date, and forest_mask key linking to the path of the pixels of interest.

    Returns
    -------
    stack_vi : array (Time,x,y)
        Array containing corrected vegetation index data
    large_scale_model : xarray (coeff: 5)
        Array containing the five coefficients of the large scale median vegetation index model
    correction_vi : xarray (Time)
        Array containing the correction terms for each date which were added to the vegetation index for its correction

    """
    
    forest_mask = import_binary_raster(dict_paths["forest_mask"])
    median_vi=[]
    for date in stack_vi.Time.data:
        vegetation_index, mask = import_masked_vi(dict_paths, date)
        try:
            median_vi += [float(vegetation_index.where(forest_mask & ~mask,drop =True).median())]
        except ValueError:
            median_vi += [0]
    median_vi = xr.DataArray(np.array(median_vi), coords=stack_vi.Time.coords)
    large_scale_model = model_vi(median_vi, median_vi==0, one_dim = True)
    predicted_median_vi = prediction_vegetation_index(large_scale_model,median_vi.Time.data)
    correction_vi = (predicted_median_vi - median_vi).where(median_vi!=0,0)
    stack_vi = stack_vi + correction_vi
    
    return stack_vi, large_scale_model, correction_vi

def correct_vi_date(vegetation_index, mask, forest_mask, large_scale_model, date, correction_vi):
    """
    Corrects single date vegetation index using large scale vegetation index median value previously computed.
    The difference between the prediction of the model and the large scale median value is used as a correction term for the vegetation index.
    This is meant to take into account changes affecting all pixels equally which are not linked with a dieback.

    Parameters
    ----------
    vegetation_index : xarray DataArray
        DataArray containing uncorrected vegetation index values
    mask : xarray DataArray
        DataArray containing mask values.
    forest_mask : xarray DataArray
        Binary array containing True if pixels are inside the region of interest.
    large_scale_model : xarray (coeff: 5)
        Array containing the five coefficients of the large scale median vegetation index model
    date : str
        Date in the format "YYYY-MM-DD"
    correction_vi : xarray (Time)
        Array containing the correction terms for each date which were added to the vegetation index for its correction

    Returns
    -------
    vegetation_index : xarray DataArray
        DataArray containing corrected vegetation index values
    correction_vi : xarray (Time)
        Array containing the correction terms for each date, with added correction term of the date used in the function.

    """
    
    if date not in correction_vi.Time:
        median_vi = vegetation_index.where(forest_mask & ~mask).median(dim=["x","y"])
        if np.isnan(median_vi):
            date_correction_vi = xr.DataArray(0, coords={"Time" : [date]},dims=["Time"])
        else:
            date_correction_vi = prediction_vegetation_index(large_scale_model,[date]) - median_vi
        correction_vi = xr.concat((correction_vi,date_correction_vi),dim = 'Time')

    vegetation_index = vegetation_index + correction_vi.sel(Time = date)
    return vegetation_index, correction_vi
