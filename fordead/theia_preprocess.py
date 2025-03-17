# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:53:50 2021

@author: Raphael Dutrieux
@author: Florian de Boissieu
"""
from datetime import timedelta
from eodag import EODataAccessGateway
from eodag.crunch import FilterProperty
import json
import numpy as np
import pandas as pd
from path import Path
import rasterio
import re
import tempfile
import time
from zipfile import ZipFile, BadZipfile
import zipfile

from fordead.import_data import retrieve_date_from_string, TileInfo
from fordead.stac.theia_collection import parse_theia_name

def get_local_maja_files(unzip_dir):
    """
    List the local maja files (zip and unzip) and 
    some of there characteristics (version, empty zip, ...)
    
    Parameters
    ----------
    unzip_dir : str
        Path of the directory with unzipped theia data. The default is None.

    Returns
    -------
    pandas.DataFrame of the files
    """

    if unzip_dir is not None:
        unzip_dir = Path(unzip_dir)
        unzip_files = unzip_dir.glob("SENTINEL*")
    else:
        unzip_files = []

    merged_files = []
    for f in unzip_files:
        merged_file =  f/"merged_scenes.json"
        if merged_file.is_file():
            with open(merged_file, "r") as ff:
                merged_list = json.load(ff)
            for v in merged_list[str(f.name)]:
                if f.name != v:
                    merged_files.append(unzip_dir/v)
    
    unzip_files.extend(merged_files)

    df = pd.DataFrame({
        "date": [retrieve_date_from_string(f) for f in unzip_files],
        "id": [re.sub(r'(.*)_[A-Z]_V[0-9]-[0-9]$', r'\1',f.stem) for f in unzip_files],
        "version": [re.sub(r".*_V([0-9]-[0-9])$", r"\1", Path(f).stem) for f in unzip_files],
        "unzip_file": [f if f.is_dir() else pd.NA for f in unzip_files],
    }) #.astype({"id": str, "version": str, "unzip_file": str})

    # identify duplicates
    df["merged"] = df["date"].duplicated(keep=False)
    merged_id = df.loc[df["merged"] & df["unzip_file"].notna(), ["id", "date"]]
    merged_id.rename(columns={"id":"merged_id"}, inplace=True)
    df = df.merge(merged_id, how="left", on="date")

    return df

def maja_search(
        tile: str,
        start_date: str,
        end_date: str = None,
        lim_perc_cloud: float = 100,
        level: str = 'LEVEL2A',
        search_timeout: int = 10,):
    """
    Download Sentinel-2 L2A data processed with MAJA processing chain from
    GEODES (CNES) plateform.

    Parameters
    ----------
    tile : str
        Name of the Sentinel-2 tile (format : "T31UFQ").
    start_date : str
        Acquisitions before this date are ignored (format : "YYYY-MM-DD") 
    end_date : str
        Acquisitions after this date are ignored (format : "YYYY-MM-DD") 
    lim_perc_cloud : int
        Maximum cloud cover (%)
    level : str, optional (see notes)
        Product level for reflectance products can be 'LEVEL2A' or 'LEVEL3A'
    search_timeout : int, optional
        Timeout in seconds for search of theia products. Default is 10.
        It updates the htttp request timeout for the search.

    Returns
    -------
    List
        Files to unzip

    Notes
    -------
    See `maja_download()` for authentication details
    """

    if level == 'LEVEL2A':
        product_type = 'S2_MSI_L2A_MAJA'
        provider = 'geodes'
        # MGRS tile has multiple formats in geodes metadata
        # - before 2024-09-25: TXXYYY
        # - after 2024-09-25: XXYYY
        # Thus both are searched
        tile_arg = [{"spaceborne:tile": "T"+re.sub(r"^T", "", tile)},
                    {"spaceborne:tile": re.sub(r"^T", "", tile)}]
    elif level == 'LEVEL3A':
        raise NotImplementedError("LEVEL3A not yet implemented, ask developers if needed.")
        # product_type = 'S2_MSI_L3A_WASP'
        # provider = 'theia'
        # tile_arg = [{"location":re.sub(r'^([0-9].*)', r'T\1', tile)}]

    if end_date is None:
        end_date = str((pd.to_datetime(start_date) + timedelta(days=1)).date())

    dag = EODataAccessGateway()
    # patch for download issue mentioned in MR !12
    # making more specific the jsonpath expression to avoid
    # multiple downloadLink path leading to NotAvailable locations
    dag.providers_config["geodes"].search.metadata_mapping["downloadLink"] = '$.assets[?(@.roles[0] == "data" & @.type == "application/zip")].href'

    # dag._prune_providers_list()

    if search_timeout is not None:
        dag.providers_config[provider].search.timeout = search_timeout

    # search products
    all_search_results = []
    for t in tile_arg:
        search_args = {
            "productType":product_type,
            "start":start_date,
            "end":end_date,
            # "cloudCover":lim_perc_cloud,
            "provider":provider
        }
        search_args.update(t)
        search_results = dag.search_all(**search_args)
            
        # issue: cloudCover criterium not working with geodes
        # TODO: actually all date duplicates should be kept,
        # to avoid merging differently dependending on cloudCover
        search_results = search_results.crunch(
            FilterProperty(dict(cloudCover=lim_perc_cloud, operator="lt"))
        )

        all_search_results.extend(search_results)

    search_results = all_search_results

    df_remote = []
    for r in search_results:
        # fix to keep a short id
        r.properties["id"] = r.properties["identifier"]
        # fix to keep same zip name as theia
        r.properties["title"] = r.properties["identifier"]
        date = retrieve_date_from_string(r.properties["identifier"])
        props = dict(
            id = re.sub(r'(.*)_[A-Z]', r'\1',r.properties["id"]),
            date = date,
            # version 4.0 converted to "4-0"
            version = re.sub(r'\.', '-', str(r.properties["versionInfo"])),
            cloud_cover = r.properties["cloudCover"],
            product = r,
        )
        # from copy import deepcopy
        # rprops = deepcopy(r.properties)
        # rprops.pop("id")
        # props.update(rprops)
        df_remote.append(props)
    if len(df_remote) == 0:
        return pd.DataFrame(dict(id=[], date=[], version=[], cloud_cover=[], product=[]))
    df_remote = pd.DataFrame(df_remote)

    return df_remote

def categorize_search(search_results, unzip_dir):
    # Merge the local file list with the remote search
    # and add column status specifying the action to take
    df_local = get_local_maja_files(unzip_dir)
    # df_columns = ['id', "date", 'zip_file', "zip_exists", 'unzip_file', "unzip_exists", "merged", "merged_id", "version"]
        
    df = df_local.merge(search_results, how="outer", on=["date","id"], suffixes=("_local", "_remote"))

    def categorize(x):
        if x["version_local"] == x["version_remote"]:
            return "up_to_date"
        elif pd.isnull(x["unzip_file"]) and (not x["merged"] or pd.isnull(x["merged"])):
            return "download"
        elif not pd.isnull(x["version_local"]) and not pd.isnull(x["version_remote"]) and x["version_local"] != x["version_remote"]:
            return "upgrade"
        else:
            return "unknown"
        
    df["status"] = df.apply(lambda x : categorize(x), axis=1)

    # extend download to duplicates if any
    df.loc[df.date.duplicated(keep=False), "status"] = df.loc[df.date.duplicated(keep=False)].groupby(by="date")["status"].transform(lambda x: "download" if (x=="download").any() else x)

    # extend upgrade to duplicates if any
    df.loc[df.date.duplicated(keep=False), "status"] = df.loc[df.date.duplicated(keep=False)].groupby(by="date")["status"].transform(lambda x: "upgrade" if (x=="upgrade").any() else x)

    return df


def maja_download(
        tile: str,
        start_date: str,
        end_date: str,
        zip_dir: str,
        unzip_dir: str,
        keep_zip: bool = False,
        lim_perc_cloud: float = 100,
        level: str = 'LEVEL2A',
        bands: list = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"],
        correction_type: str = "FRE",
        search_timeout: int = 10,
        upgrade: bool = True,
        dry_run: bool = True,
        retry: int = 10,
        wait: int = 5
):
    """
    Search, download, unzip and merge duplicates of Sentinel-2 L2A data processed with
    MAJA processing chain from GEODES (CNES) plateform.

    Parameters
    ----------
    tile : str
        Name of the Sentinel-2 tile (format : "T31UFQ").
    start_date : str
        Acquisitions before this date are ignored (format : "YYYY-MM-DD") 
    end_date : str
        Acquisitions after this date are ignored (format : "YYYY-MM-DD") 
    zip_dir : str
        Directory where THEIA data is downloaded
    unzip_dir : str
        Directory where THEIA data is unzipped
    keep_zip : bool
        If False, removes the zip file after extraction
    lim_perc_cloud : int
        Maximum cloud cover (%)
    level : str, optional (see notes)
        Product level for reflectance products, can be 'LEVEL2A' or 'LEVEL3A'
    search_timeout : int, optional
        Timeout in seconds for search of theia products. Default is 10.
        It updates the htttp request timeout for the search.
    bands : list, optional
        List of bands to download. Default is ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"]
    correction_type : str, optional
        Correction type for radiometric correction, can be "SRE" (Surface Reflectance) or "FRE" (Falt Reflectance) for
        LEVEL2A . Default is "FRE".
    upgrade : bool, optional
        If True, checks product version and upgrades if needed
    dry_run : bool, optional
        If True, do not download or unzip products
    retry : int, optional
        Number of times to retry a failed download
    wait : int, optional
        Number of minutes to wait between retries

    Returns
    -------
    tuple
        downloaded, unzipped

    Notes
    -------
    GEODES is the plateform used to download Sentinel-2 data.
    In order to have download access to that plateform, you will
    need to create an account at https://geodes.cnes.fr/ and create
    an API key on that account.

    The API key should then be specified in the section `geodes`
    of the EODAG config file "$HOME/.config/eodag/eodag.yaml".
    It should look like this:
    ```yaml
    geodes:
        priority: # Lower value means lower priority (Default: 0)
        search: # Search parameters configuration
        auth:
            credentials:
                apikey: "write your api key here"
        download:
            extract:
            output_dir:
    ```

    If $HOME/.config/eodag/eodag.yaml does not exists,
    run the following in a python session, it should create the file:
    ```python
    from path import Path
    from eodag import EODataAccessGateway
    
    # creates the default config file at first call
    EODataAccessGateway()

    # prints the path to the default config file, makes sure it exists
    config_file = Path("~/.config/eodag/eodag.yml").expand()
    print(config_file)
    assert config_file.exists()
    ```

    If the geodes section does not exist in the config file,
    it must be that the config file template is not up to date
    with EODAG v3+. In order to fix it, rename the file
    "$HOME/.config/eodag/eodag.yaml" to "$HOME/.config/eodag/eodag.yaml.old"
    and create a new with the above code.

    If there are special characters in your API key,
    make sure to add double quotes around it.
    """

    if level == "LEVEL3A" : correction_type = "FRC"

    search = maja_search(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        lim_perc_cloud=lim_perc_cloud,
        level=level,
        search_timeout=search_timeout)
    
    df = categorize_search(search, unzip_dir)

    # merged products set to upgrade by extension may not be part of search
    # because of cloud cover limit, thus they are added a-posteriori
    if any(df["status"].isin(["upgrade", "download"]) & df["product"].isnull()):
        df_issue = df[df['status'].isin(["upgrade", "download"]) & df['product'].isnull()]
        message = '\n'.join(df_issue['id'])
        print("Products meant to be upgraded and merged but not included in the search results are added:"
                           f"\n{message}")
        for d in df_issue["date"]:
            search_sup = maja_search(
                tile=tile,
                start_date=d,
                end_date = str((pd.to_datetime(d) + timedelta(days=1)).date()),
                lim_perc_cloud=100,
                level=level,
                search_timeout=search_timeout)
            search_sup = search_sup[~search_sup["id"].isin(search["id"])]
            message = '\n'.join(search_sup["id"])
            print(f"Product added to search results: {message}")
            search = pd.concat([search, search_sup]).sort_values("date", ignore_index=True) # add to search

        df = categorize_search(search, unzip_dir)
    
    # keep only search results
    df = df[df["product"].notna()]

    if upgrade:
        df_download = df[df["status"].isin(["upgrade", "download", "badzip"])]
    else:
        df_download = df[df["status"].isin(["download", "badzip"])]

    if len(df_download) == 0:
        print("Already up-to-date: nothing to download.")
        return [], []

    unzipped = df[df["status"]=="up-to-date"]
    to_upgrade = df_download[df_download["status"]=="upgrade"]
    to_download = df_download[df_download["status"].isin(["download", "badzip"])]
    
    if len(unzipped):
        unzipped_str = '\n'.join(unzipped["id"])
        print(f"Products up-to-date ({len(unzipped)}):\n{unzipped_str}\n")
    if len(to_upgrade):
        to_upgrade_str = '\n'.join(to_upgrade.apply(lambda x: f"{x['id']} ({x['version_local']} -> {x['version_remote']})", axis=1))
        print(f"Products to upgrade ({len(to_upgrade)}):\n{to_upgrade_str}\n")
    if len(to_download):
        to_download_str = '\n'.join(to_download.apply(lambda x: f"{x['id']} ({x['version_remote']})", axis=1))
        print(f"Products to download ({len(to_download)}):\n{to_download_str}\n")


    downloaded = []
    unzipped = []
    if dry_run:
        print("Dry run, nothing has been downloaded")
    else:
        for r in df_download.itertuples():
            # 1. download the zip file if not already there
            # 2. extract the zip file
            with tempfile.TemporaryDirectory(dir=zip_dir) as tmpdir:
                zip_file = zip_dir.glob(r.id + "*.zip")
                if len(zip_file) > 0 and check_zip(zip_file[0]):
                    zip_file = zip_file[0]
                else:
                    done = False
                    trials = 0
                    while not done:
                        try:
                            tmpzip_file = r.product.download(output_dir=tmpdir, extract=False)
                            done = True
                        except Exception as e:
                            print(e)
                            print("Failed to download: ", r.id)
                            if trials==retry:
                                raise RuntimeError("\nRetry limit reached.\n")
                            else:
                                trials += 1
                                print(f"\nRetries in {wait*trials} minutes {trials}/{retry} ...\n")
                                time.sleep(wait*trials*60)
                                
                    zip_file = Path(tmpzip_file)
                    if keep_zip:
                        zip_file = zip_file.move(zip_dir / zip_file.name)

                # remove old unzip if already there
                if pd.notna(r.unzip_file):
                    try:
                        print("Removing old unzip: ", r.unzip_file)
                        r.unzip_file.rmtree()
                    except Exception as e:
                        print("Failed to remove", r.unzip_file)
                        raise e
                # extract zip file
                tmpunzip_file = s2_unzip(zip_file, tmpdir, bands, correction_type)
                unzip_file = tmpunzip_file.move(unzip_dir / Path(tmpunzip_file).name)

            downloaded.append(unzip_file)
            unzipped.append(unzip_file)
        # TODO merge only the results of search
        merge_same_date(bands, unzip_dir, correction_type=correction_type)
    
    return downloaded, unzipped

def check_zip(x):
    """
    Check if a zip file is empty or BadZipfile

    Returns False if the zip file is empty or BadZipfile, True otherwise
    """
    if Path(x).size == 22:
        return False
    try:
        ZipFile(x).namelist()
    except BadZipfile:
        return False

    return True


BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
              'CLMR1', 'CLMR2', 'EDGR1', 'EDGR2', 'SATR1', 'SATR2',
              "DTS1","DTS2","FLG1","FLG2","WGT1","WGT2"]

BAND_FILE_PATTERNS = dict(zip(BAND_NAMES,
    ['B2.tif', 'B3.tif', 'B4.tif', 'B5.tif', 'B6.tif', 'B7.tif',
     'B8.tif', 'B8A.tif', 'B11.tif', 'B12.tif', '_CLM_R1.tif', 
     '_CLM_R2.tif', '_EDG_R1.tif', '_EDG_R2.tif', '_SAT_R1.tif',
     '_SAT_R2.tif',"_DTS_R1.tif","_DTS_R2.tif","_FLG_R1.tif",
     "_FLG_R2.tif","_WGT_R1.tif","_WGT_R2.tif"]))

def theia_bands(correction_type):
    """
    Returns dict of expected theia bands for the specified correction type

    Parameters
    ----------
    correction_type : str
        Chosen correction type (SRE or FRE for LEVEL2A data, FRC for LEVEL3A)

    Returns
    -------
    dict
        Theia band patterns for the specified correction type
    """
    bands = {k:re.sub(r'^(B[0-9]+[A]*.tif)$', "_"+correction_type+'_\\1', v) for k, v in BAND_FILE_PATTERNS.items()}
    return bands


def s2_unzip(s2zipfile, out_dir, bands, correction_type):
    """
    Unzips Flat REflectance data from zipped theia data, then empties the zip file

    Parameters
    ----------
    s2zipfile : str
        Path of the zip file containing a Sentinel-2 acquisition downloaded from theia.
    out_dir : str
        Directory where data is extracted
    bands : list of str
        List of bands to extracted (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, as well as CLMR2, CLMR2, EDGR1, EDGR2, SATR1, SATR2 for LEVEL2A data, and DTS1, DTS2, FLG1, FLG2, WGT1, WGT2 for LEVEL3A)
    correction_type : str
        Chosen correction type (SRE or FRE for LEVEL2A data, FRC for LEVEL3A)

    Returns
    -------
    None.

    """
    s2zipfile = Path(s2zipfile).expand()
    out_dir = Path(out_dir).expand()
    corrBand = theia_bands(correction_type)
  
    
    try:
        with ZipFile(s2zipfile, 'r') as zipObj:
            root_dir, = zipfile.Path(zipObj).iterdir()
            if (out_dir / root_dir.name).exists():
                print(f'Files already extracted: {out_dir / root_dir.name}')
            else:
                print(f'Extracting files from {s2zipfile} to {out_dir / root_dir.name}')
                with tempfile.TemporaryDirectory(dir=out_dir) as tempdir:
                    tmpdir = Path(tempdir)
                    listOfFileNames = zipObj.namelist()
                    for fileName in listOfFileNames:
                            for i in bands:
                                if fileName.endswith(corrBand[i]): #Si le nom de fichier finit par ce qui correspond à la bande
                                    zipObj.extract(fileName, tmpdir)
                    (tmpdir / root_dir.name).move(out_dir)
            return out_dir / root_dir.name    
    except BadZipfile:
        print(f"Bad zip file, removing file: {s2zipfile}")
        s2zipfile.remove()

def unzip_theia(bands, zip_dir, out_dir, empty_zip, correction_type):
    """
    Unzips zipped theia data, then empties the zip file

    Parameters
    ----------
    bands : list of str
        List of bands to extracted (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, as well as CLMR2, CLMR2, EDGR1, EDGR2, SATR1, SATR2 for LEVEL2A data, and DTS1, DTS2, FLG1, FLG2, WGT1, WGT2 for LEVEL3A)
    zip_dir : str or list
        Directory where zip files containing theia data are stored
        or list of zip files
    out_dir : str
        Directory where data is extracted
    empty_zip : bool
        If True, the zip file is emptied as a way to save space
    correction_type : str
        Chosen correction type (SRE or FRE for LEVEL2A data, FRC for LEVEL3A)


    Returns
    -------
    None.
    

    """
    
    out_dir = Path(out_dir).expand()
    if isinstance(zip_dir, list):
        zipList = [Path(f).expand() for f in zip_dir]
    else:
        zip_dir = Path(zip_dir).expand()
        zipList=zip_dir.glob("*.zip")
    
    out_dir.mkdir_p()
    
    for zipfile in zipList:
        s2_unzip(zipfile, out_dir, bands, correction_type)
        if zipfile.exists() and (len(ZipFile(zipfile).namelist()) != 0) and empty_zip:
            print("Replaces by empty zip: " + zipfile)
            zipfile.remove()
            zipObj = ZipFile(zipfile, 'w')
            zipObj.close()

def delete_empty_zip(zipped_dir, unzipped_dir):
    """
    Deletes empty zip files if the unzipped directory is missing

    Parameters
    ----------
    zipped_dir : str
        Directory where zip files containing theia data are stored.
    unzipped_dir : str
        Directory where unzipped data containing theia data are stored

    Returns
    -------
    None.

    """

    tile = TileInfo(unzipped_dir)
    for i in range(2):
        tile.getdict_datepaths("zipped",zipped_dir)
        tile.getdict_datepaths("unzipped",unzipped_dir)
        for date in tile.paths["zipped"]:
            if date not in tile.paths["unzipped"]: # this excludes duplicates of the same date but different scene ID, it is wanted...
                try:
                    if (len(ZipFile(tile.paths["zipped"][date]).namelist()) == 0):
                        print("Zip file is empty and unzipped directory not found : zip file removed " + str(tile.paths["zipped"][date]))
                        Path(tile.paths["zipped"][date]).remove()
                except BadZipfile:
                    print("Bad zip file, removing file: {}".format(tile.paths["zipped"][date]))
                    Path(tile.paths["zipped"][date]).remove()

def merge_same_date(bands, out_dir, correction_type):
    """
    Merges data from two theia directories if they share the same date of acquisition

    Parameters
    ----------
    bands : list of str
        List of bands to extracted (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, CLMR2).
    out_dir : str
        Directory where unzipped data containing theia data are stored.

    Returns
    -------
    None.

    """

    if not all([b.startswith("B") or b.startswith("CLMR") for b in bands]):
        raise NotImplementedError("Only bands Bxx and CLMR[1-2] are supported for merge.")

    out_dir = Path(out_dir).expand()
    SenPathGen = out_dir.glob("SEN*")
    # SenPathList=glob(out_dir+"/SEN*")
    SenDateList = [] #Création dictionnaire avec date : chemin du fichier
    SenPathList = []
    for SenPath in SenPathGen:
        # SenDate = SenPath.split('_')[1].split('-')[0]
        SenDate = retrieve_date_from_string(SenPath)
        SenDateList+=[SenDate]
        SenPathList += [SenPath]
    # tile = TileInfo(out_dir)
    # tile.getdict_datepaths("SENTINEL",tile.data_directory)

    merged_file_name = "merged_scenes.json"
    for date in np.unique(np.array(SenDateList)):
        if np.sum(np.array(SenDateList)==date)>1:
            print("Duplicates detected at date : " + date)
            Doublons=np.array(SenPathList)[np.array(SenDateList)==date]
            Doublons = [Path(f) for f in Doublons]
            Doublons.sort()
            # check if already merged
            for doublon in Doublons:
                if (doublon / merged_file_name).exists():
                    raise Exception("Duplicate "+doublon.name+" already merged.")

            with tempfile.TemporaryDirectory(dir=out_dir) as tempdir:
                tmpdir = Path(tempdir)

                # mosaic bands of the same date
                corr_band = theia_bands(correction_type)
                for band in bands:
                    print("Mosaicing band : " + band)
                    for i, doublon in enumerate(Doublons):
                        if band.startswith("CLMR"):
                            filename = Path("MASKS") / (doublon.name + corr_band[band])
                            band_path = doublon / filename
                            na_value=0
                        elif band.startswith("B"):
                            filename= doublon.name + corr_band[band]
                            band_path = doublon / filename
                            na_value=-10000
                        else:
                            raise NotImplementedError("Merge not implemented for band " + band)

                        if i==0:
                            output_filename = filename
                            output_dir = doublon
                            tmp_path = tmpdir / output_filename
                            tmp_path.parent.mkdir_p()

                        
                        with rasterio.open(band_path) as RasterBand:
                            if i==0:
                                merged_band=RasterBand.read(1)
                                profile=RasterBand.profile
                            else:
                                added_band=RasterBand.read(1)
                                merged_band[added_band!=na_value]=added_band[added_band!=na_value]
                    
                    with rasterio.open(tmp_path, 'w', **profile) as dst:
                        dst.write(merged_band,indexes=1)
                
                # add a file that specifies the scene was merged with others
                with open(tmpdir / merged_file_name, "w") as f:
                    json.dump({output_dir.name:[d.name for d in Doublons]},f, indent=4)
                    
                # remove merged scenes
                for doublon in Doublons:
                    doublon.rmtree()
                
                # move merged scene to original directory
                tmpdir.move(output_dir)

def patch_merged_scenes(zip_dir, unzip_dir, dry_run=True):
    """
    Adds a merged_scenes.json file to theia directories
    if a zip file exists at the same date but its unzip
    directory is missing


    Parameters
    ----------
    zip_dir : str
        Directory where theia zip files are stored (e.g. "/home/fordead/data/Zipped/T31TGM")
    unzip_dir : str
        Directory where theia unzipped directories are stored (e.g. "/home/fordead/data/Unzipped/T31TGM")
    dry_run : bool
        If True, merged_scenes.json are not written

    Returns
    -------
    List
        List of edited files merged_scenes.json
    """

    unzip_dir = Path(unzip_dir)
    unzip_files = unzip_dir.glob("SENTINEL*")

    zip_dir = Path(zip_dir)
    zip_files = zip_dir.glob("SENTINEL*.zip")

    df_unzip = pd.DataFrame({
        "date": [retrieve_date_from_string(f) for f in unzip_files],
        "id": [re.sub(r'(.*)_[A-Z]_V[0-9]-[0-9]$', r'\1',f.stem) for f in unzip_files],
        "version": [re.sub(r".*_V([0-9]-[0-9])$", r"\1", Path(f).stem) for f in unzip_files],
        "unzip_file": unzip_files,
    })

    df_zip = pd.DataFrame({
        "date": [retrieve_date_from_string(f) for f in zip_files],
        "id": [re.sub(r'(.*)_[A-Z]$', r'\1',f.stem) for f in zip_files],
        "zip_file": zip_files,
        })
    
    df = df_zip.merge(df_unzip, how="outer", on=["date", "id"], left_index=False, right_index=False)
    
    df["merged"] = df["date"].duplicated(keep=False)
    merged_id = df.loc[df["merged"] & df["unzip_file"].notna()]
    merged_id.rename(columns={"id":"merged_id"}, inplace=True)
    df = df.merge(merged_id[["merged_id", "date"]], how="left", on="date")
    merged_scenes_files = []
    for r in merged_id.itertuples():
        if not (r.unzip_file / "merged_scenes.json").exists():
            merged_list = []
            for id in df.loc[df["merged_id"] == r.merged_id].id.to_list():
                filename = id + re.sub(r'.*(_[A-Z]_V[0-9]-[0-9])$', r'\1',r.unzip_file.stem)
                merged_list.append(filename)

            merged = {str(r.unzip_file) : [str(f) for f in merged_list]}
            print("merged scenes: ", merged)
            if dry_run:
                print("WARNING: dry run, not writing merged_scenes.json in: ", r.unzip_file)
            else:
                print("writing merged_scenes.json in: ", r.unzip_file)
                with open(r.unzip_file / "merged_scenes.json", "w") as f:
                    json.dump(merged, f, indent=4)

            merged_scenes_files.append(r.unzip_file / "merged_scenes.json")
    
    return merged_scenes_files

            