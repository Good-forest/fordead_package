# -*- coding: utf-8 -*-
"""
Module with the theia preprocess function and a corresponding command line.
"""

import click
from path import Path
from datetime import date
from fordead.theia_preprocess import unzip_theia, merge_same_date, delete_empty_zip, theia_download


@click.command(name='theia_preprocess')
@click.option("-i", "--zipped_directory", type = str, help = "Path of the directory with zipped theia data")
@click.option("-o", "--unzipped_directory", type = str, help = "Path of the output directory")
@click.option("-t", "--tiles", multiple=True, help = "Name of the tiles to be downloaded (format : T31UFQ)")
@click.option("-u", "--login_theia", type = str, help = "Login of your theia account")
@click.option("-p", "--password_theia", type = str, help = "Password of your theia account")
@click.option("-l", '--level', type = click.Choice(['LEVEL1C', 'LEVEL2A', 'LEVEL3A'], case_sensitive=False),  help='Product level for reflectance products', default='LEVEL2A', show_default=True)
@click.option("-s", "--start_date", type = str, default = "2015-06-23",help = "start date, fmt('2015-12-22')", show_default=True)
@click.option("-e", "--end_date", type = str, default = None, help = "end date, fmt('2015-12-22'). If None, the current date is used.", show_default=True)
@click.option("-n", "--lim_perc_cloud", type = int,default = 50, help = "Maximum cloudiness in SENTINEL dates downloaded (%)", show_default=True)
@click.option("-b", "--bands", multiple=True, default=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"],help = "List of bands to extracted (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, as well as CLMR2, CLMR2, EDGR1, EDGR2, SATR1, SATR2 for LEVEL2A data, and DTS1, DTS2, FLG1, FLG2, WGT1, WGT2 for LEVEL3A)", show_default=True)
@click.option("-c", '--correction_type', type = click.Choice(['SRE', 'FRE', 'FRC'], case_sensitive=False),  help='Chosen correction type (SRE or FRE for LEVEL2A data, FRC for LEVEL3A)', default='FRE', show_default=True)
@click.option("--empty_zip",  is_flag=True, help = "If True, the zip files are emptied as a way to save space.", show_default=True)
@click.option("--retry", type=int, default=3, help = "Number of retries when downloading data", show_default=True)
@click.option("--wait", type=int, default=300, help = "Wait time between retries in seconds", show_default=True)
@click.option("--search_timeout", type=int, default=10, help = "Search time out in seconds", show_default=True)
def cli_theia_preprocess(zipped_directory, unzipped_directory, tiles, login_theia, password_theia,
                         level, start_date = "2015-06-23", end_date = "2023-06-23", lim_perc_cloud = 50,
                         bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"],
                         correction_type = "FRE", empty_zip = False, retry = 3, wait=300, search_timeout=10):
    """
    Automatically downloads all Sentinel-2 data from THEIA between two dates under a cloudiness threshold. Then this data is unzipped, keeping only chosen bands from Flat REflectance data, and zip files can be emptied as a way to save storage space.
    Finally, if two Sentinel-2 directories come from the same acquisition date, they are merged by replacing no data pixels from one directory with pixels with data in the other, before removing the latter directory.

    \f

    """
    
    theia_preprocess(
        zipped_directory=zipped_directory,
        unzipped_directory=unzipped_directory,
        tiles=tiles,
        login_theia=login_theia,
        password_theia=password_theia,
        level=level,
        start_date=start_date,
        end_date=end_date,
        lim_perc_cloud=lim_perc_cloud,
        bands=bands,
        correction_type=correction_type,
        empty_zip=empty_zip,
        retry=retry,
        wait=wait,
        search_timeout=search_timeout)

def theia_preprocess(zipped_directory, unzipped_directory, tiles, login_theia=None, password_theia=None,
                     level = "LEVEL2A", start_date = "2015-06-23", end_date = None, lim_perc_cloud = 50,
                     bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"], 
                     correction_type = "FRE", empty_zip = False, retry = 10, wait=300, search_timeout=10):
    """
    Download Sentinel-2 zip files from THEIA portal, 
    extract band files and eventually merge tile+date duplicates.
    
    Scenes can be filtered on cloud coverage, and zip files
    can be emptied as a way to save storage space while avoiding
    downloading the same data twice.
    Finally, if two Sentinel-2 directories come from the same acquisition date,
    they are merged by replacing sequentially the valid pixels of each duplicate.

    Parameters
    ----------
    zipped_directory : str
        Path of the directory with zipped theia data.
    unzipped_directory : str
        Path of the output directory.
    tiles : list of str
        Name of the tiles to be downloaded (format : T31UFQ)
    login_theia : str
        Login of your theia account.
    password_theia : str
        Password of your theia account.
    level : str
        Product level for reflectance products, can be 'LEVEL1C', 'LEVEL2A' or 'LEVEL3A'
    start_date : str, optional
        start date, fmt('2015-12-22'). The default is "2015-06-23".
    end_date : str, optional
        end date, fmt('2015-12-22'). The default is None.
    lim_perc_cloud : int, optional
        Maximum cloudiness in SENTINEL dates downloaded (%). The default is 50.
    bands : list of str
        List of bands to extracted (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, as well as CLMR2, CLMR2, EDGR1, EDGR2, SATR1, SATR2 for LEVEL2A data, and DTS1, DTS2, FLG1, FLG2, WGT1, WGT2 for LEVEL3A). The default is ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLMR2"].
    correction_type : str
        Chosen correction type (SRE or FRE for LEVEL2A data, FRC for LEVEL3A)
    empty_zip : bool, optional
        If True, the zip files are emptied as a way to save space. The default is False.
    retry : int, optional
        Number of retries if download fails. The default is 10.
    wait : int, optional
        Wait time between retries in seconds. The default is 300.
    search_timeout : int, optional
        Search timeout in seconds. The default is 10.
    
    Returns
    -------
    None.

    """
    
    if level == "LEVEL3A" : correction_type = "FRC"
    
    zipped_directory = Path(zipped_directory).expanduser().realpath().mkdir_p()
    unzipped_directory = Path(unzipped_directory).expanduser().realpath().mkdir_p()
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    
    for tile in tiles:
        print("\n Downloading THEIA data for tile " + tile + "\n")
        tile_zip_dir = (zipped_directory / tile).mkdir_p()   
        tile_unzip_dir = (unzipped_directory / tile).mkdir_p()
        

        delete_empty_zip(tile_zip_dir, tile_unzip_dir) #Deletes empty zip files if the unzipped directory is missing
                
        to_unzip = theia_download(tile, start_date, end_date, tile_zip_dir,
                    lim_perc_cloud, login_theia, password_theia, level, 
                    tile_unzip_dir, retry=retry, wait=wait, search_timeout=search_timeout)

        unzip_theia(bands, to_unzip, tile_unzip_dir, empty_zip, correction_type)
        merge_same_date(bands, tile_unzip_dir)

if __name__ == '__main__':
    # start_time_debut = time.time()
    cli_theia_preprocess()
    # print("Calcul des masques et du CRSWIR : %s secondes ---" % (time.time() - start_time_debut))

