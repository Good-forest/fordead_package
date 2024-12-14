# Principle

This code shows how to compute the 5 harmonic parameters from time series of the radiometric index. 
It needs **OTBTF >= 3.0** to work.

The idea is to generate a **TensorFlow SavedModel**, which is a computational graph serialized using Google Protobuf, 
and use it with the `TensorflowModelServe` application of the **OTBTF remote module of the Orfeo ToolBox**. TensorFlow 
can run on many computing hardware, including **GPUs** and **CPUs**. Regarding this matter can use OTBTF docker images 
suited for the hardware of your choice.

# Detailed steps

We assume that the radiometric index of interest in computed for each image in a mono-band raster in the `$DATADIR`
folder.

### Enter the docker image

In the following example we use the **otbtf3.0-gpu** docker image.

```bash
docker run --gpus all -ti -v $DATADIR:$DATADIR mdl4eo/otbtf3.0:gpu
```
### Generate SavedModel

The implemented model has the following inputs:
- 1 input image "_ts_", for the vegetation index time series, 
- 1 constant input vector "_dates_", containing the dates in doy format.
Note that the number of components in "_ts_" and "_dates_" must be the same.

The following command generates the Tensorflow SavedModel, implementing the computation of the 5 harmonic parameters.
```bash
python generate_fordead5params_model.py
```
Now the `fordead5params_model` directory has been created in the current path.

### Run the model

The principle is the following:
```bash
# otbcli_TensorflowModelServe                  \
# -source1.il crswir_1.tif ... crswir_N.tif    \ # list of radiometric index, sorted by ascending date
# -model.userplaceholders dates=(1, ..., 547)  \ # list of ascending dates in doy format
# -model.dir fordeadparams_model/              \ # directory containing the SavedModel 
# -model.fullyconv on                          \ # the model reads image chunks  
# -out coefs.tif                                 # output raster
```

We can speed up the process tuning the `-optim` parameters group.
For instance adding `-optim.tilesizex 10000` should be fine and optimize access for S2 images strips. 
You can then adjust the `-optim.tilesizey`  value depending on the available memory budget.

Here are the command lines to run our example:
1. List the available rasters:
```bash
files=$(find $DATADIR/VegetationIndex/ -type f | sort)
```
2. Extract the dates:
```bash
dates=$(echo $(echo "$files" | rev | cut -f1 -d_ | rev | cut -f1 -d. | xargs -i date -d {} +%j) | sed "s/ /.0,/g")
```
3. Run the model:

```bash
otbcli_TensorflowModelServe -source1.il $files -model.userplaceholders "dates=($dates)" \
-model.dir fordead5params_model/ -model.fullyconv on -out coefs.tif -optim.tilesizex 10000
```
You should have the following output:
```
2021-11-23 15:12:04 (INFO) TensorflowModelServe: Default RAM limit for OTB is 256 MB
2021-11-23 15:12:04 (INFO) TensorflowModelServe: GDAL maximum cache size is 1599 MB
2021-11-23 15:12:04 (INFO) TensorflowModelServe: OTB will use at most 8 threads
2021-11-23 15:12:04.676909: I tensorflow/cc/saved_model/reader.cc:32] Reading SavedModel from: fordeadparams_model/
2021-11-23 15:12:04.679769: I tensorflow/cc/saved_model/reader.cc:55] Reading meta graph with tags { serve }
2021-11-23 15:12:04.679823: I tensorflow/cc/saved_model/reader.cc:93] Reading SavedModel debug info (if present) from: fordeadparams_model/
2021-11-23 15:12:04.679910: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-23 15:12:04.712354: I tensorflow/cc/saved_model/loader.cc:206] Restoring SavedModel bundle.
2021-11-23 15:12:04.714273: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2899885000 Hz
2021-11-23 15:12:04.734751: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2021-11-23 15:12:04.734802: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
2021-11-23 15:12:04.735035: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2021-11-23 15:12:04.735104: I tensorflow/cc/saved_model/loader.cc:190] Running initialization op on SavedModel bundle at path: fordeadparams_model/
2021-11-23 15:12:04.744897: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2021-11-23 15:12:04.744926: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
2021-11-23 15:12:04.745079: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2021-11-23 15:12:04.745159: I tensorflow/cc/saved_model/loader.cc:277] SavedModel load for tags { serve }; Status: success: OK. Took 68251 microseconds.
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Source info :
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Receptive field  : [1, 1]
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Placeholder name : 
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Output spacing ratio: 1
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Using placeholder dates with Tensor shape is {92} data type is 1 (float)
2021-11-23 15:12:05 (INFO) TensorflowModelServe: The TensorFlow model is used in fully convolutional mode
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Output field of expression: [1, 1]
2021-11-23 15:12:05 (INFO) TensorflowModelServe: Force tiling with squared tiles of [10000, 16]
2021-11-23 15:12:05 (INFO): Estimated memory for full processing: 143.675MB (avail.: 256 MB), optimal image partitioning: 1 blocks
2021-11-23 15:12:05 (INFO): File coefs.tif will be written in 1 blocks of 432x300 pixels
Writing coefs.tif...: 100% [**************************************************] (0s)

```
