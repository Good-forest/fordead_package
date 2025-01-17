# v1.11.0
## Change
- download MAJA archives from GEODES instead of THEIA:
  - rename `theia_download` to `maja_download`: more appropriate
  - unzip and merge are now done inside `maja_download`, see new args
  - add arg `upgrade` to check for MAJA version upgrade
  - add arg `dry_run` in order to print actions without doing them
  - remove args `login` and `password`: not a good practice
  - remove args `retry` and `wait`, should not be necessary anymore
  - add download tests
- remove function `delete_empty_zip`
- dependency constraints:
  - remove on python version
  - path >=17
  - eodag >= 3

## Fix
- issue #37: only one point extracted in extract_reflectance

# v1.10.1
## Add
- new `theiastac` provider for calibration/validation: extract values from
  remote STAC collection sentinel-2-l2a-theia of 
  https://stacapi-cdos.apps.okd.crocc.meso.umontpellier.fr
  to avoid downloading whole tiles. See docs/examples/calibration_validation.py

## Change
- change the output format of `extract_results` for better understanding
  (dates corresponding date indexes, renamed and split files)
- dependency to `EODAG >= 3` to solve https://github.com/CS-SI/eodag/issues/1123
- remove constraint on `numpy` version (related to stackstac new release https://github.com/gjoseph92/stackstac/issues/250)
- add constraint `python < 3.12` due to vscode issue with debugpy: see https://github.com/microsoft/vscode-python-debugger/issues/498

## Fix
- points out of the tile bbox in extract_raster_values
- removed wrong scale factor for VegetationIndex in `get_tile_collection`

# v1.10.0
## Add
- function `step5_export_results.extract_results` to extract the results for points of interest
- in validation module, provider `'theiastac'` to original `['theia', 'planetary']`

## Change
- in `extract_raster_values`:
  - switch the args order of inputs to reflect the usual signature of such a function
  - add possibility to input an xarray to `extract_raster_values`
  - add arguments with default values `chunksize=512` `dropna=True` and `dtype=int`
- `sentinel_source` is now case insensitive e.g. 'Theia', 'THEIA' and 'theia' are accepted.

## Fix
- slightly different results with same data in CalVal and production (issue #35) : 
  sometimes the pixel next to the one focused was extracted by `extract_relflectance`
  due to a shift of res/2 of the raster spatial coordinates.


# v1.9.1
## Add
- args `chunsize` and `by_chunk` to `extract_raster_values`: 
  processing by chunk to avoid loosing already downloaded when interrupted
- tests for results_visualsation

## Change
- update dependency version to eodag >= 3
- default chunsize in extract_raster_values: 512 (instead of 1024)

## Fix
- in extract_raster_values: possible duplicated {date, id_point, band}, usually due to splitted scenes --> averaging.
- result_visualisation (int index requested for xr.DataArray.isel)

# v1.9.0
## Add
- argument `start_date` to `process_tile` and `compute_masked_vegetationindex`. It defines the first date of processing.
  __Warning:__ it is retrocompatible with 1.8 (without recomputation), if and only if start_date=="2015-01-01" (default value).
  Otherwise, the time series is considered as different and the whole workflow (vegetation indices, masks, model, etc.) 
  is recomputed overwriting the existing results.
  
# v1.8.8
## Fix
- `harmonize_sen2cor_offet` --> `harmonize_sen2cor_offset`
- error message when data is insufficient for training in `train_model_from_dataframe`
- path version in requirement (< 17)
- issue with stac-geoparquet > 3.2 https://github.com/stac-utils/stac-geoparquet/issues/76

# v1.8.7
## Change
- example full_test_script.py replaced by dieback_detection.py and calibration_validation.py

## Add
- CI tests for CLI

## Fix
- Typo bug in BadZip print of theia_download
- several bugs in CLI (issue [#31](https://gitlab.com/fordead/fordead_package/-/issues/31))

# v1.8.6
## Add
- message with final count of retries in function `theia_download`.
  These retry strategy should be removed when EODAG released,
  as no retries were observed since EODAG fix.
- arg `search_timeout` in order avoid timeouts for THEIA search requests

## Fix
- `numpy < 2` inroder to avoid stackstac issue [#250](https://github.com/gjoseph92/stackstac/issues/250)
- fix issue [#27] for THEIA request with `EODAG` [#1123](https://github.com/CS-SI/eodag/issues/1123) (waiting for `EODAG 3.0`),
- some `is_dir` issue, maybe due to `path` version (to be checked)

# v1.8.5

## Fix
* issue #27 for theia_download

# v1.8.4

## Add
* function `process_tiles` in `fordead.cli.cli_process_tiles.py` with full workflow (steps 1 to 5): can be called as a function or with the CLI `fordead process_tiles`. It includes the saving of input parameters in a .json file and logging of processing time in .log file
* install and processing tests for linux and windows
* arg `retry` in `theia_preprocess` in order to retry download in case of failure
* retries (default 5 retries with 1 sec. wait between each) in case of failure for the STAC reader (e.g. for Planetary Computer)

## Changes
* setup.py --> pyproject.py with automatic package version numbering
* replaced progress prints by progress bars in step 1 and step 3
* update a few elements of documentation (e.g. print default values in CLI)

## Fix
* since versions 1.8*, a bug was introduced in `theia_preprocess` for the detection of merged scenes in the download section, see issue #25. It was fixed for the case of `empty_zip=True`. For the case for `empty_zip=False`, the function might try to re-download the scenes with duplicated dates and try to merge them again.

# v1.8.3

## Change
* the downloader of theia data is now run with eodag.
  The login and password can now be set in a configuration
  file instead of writing them as arguments,
  see [theia_preprocess](https://fordead.gitlab.io/fordead_package/docs/user_guides/english/other_tools/theia_preprocess)

# v1.8.2

## Fix
* Sen2Cor offset issue #21
* duplicates from planetary computer #20
* doc typo #19
* systematic warning from stackstac #18

## Change
* accelerate points extraction by \~x30 using xarray of the whole collection
* change a few internal function signatures, removing useless arguments (`get_already_extracted`, `extract_raster_values`)

## Add
* test in CI

# v1.8.1

## Add
* Add CI for windows

## Fix
* remove a few warnings
  