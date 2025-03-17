
from fordead.theia_preprocess import maja_download, maja_search, get_local_maja_files

def test_maja_search():
    tile = "T31TGM"

    start_date = "2024-09-26"
    end_date = "2024-10-30"
    df = maja_search(tile, start_date, end_date)
    assert all(df.version == "4-0")
    
    start_date = "2024-09-20"
    end_date = "2024-09-26"
    df = maja_search(tile, start_date, end_date)
    assert df.shape[0] == 2

    # test for search with empty results
    start_date = "2024-09-23"
    end_date = "2024-09-24"
    df = maja_search(tile, start_date, end_date)
    assert df.empty

def test_download(output_dir):
    zip_dir = (output_dir / "download" / "zip").rmtree_p().makedirs_p()
    unzip_dir = (output_dir / "download" / "unzip").rmtree_p().makedirs_p()

    # 31TGM 2018-08-11 is duplicate with cloud_cover [41,52]
    # 31TGK 2020-05-22 is duplicate with cloud_cover [11,30]
    tile = "T31TGK"
    start_date = "2020-05-22"
    end_date = "2020-05-23"
    bands=["B2", "B3", "CLMR2", "CLMR1"]
    cloud_min = 20
    cloud_max = 40

    # # 31TGM 2017-06-19 is duplicate with cloud_cover [3, 4]
    # tile = "T31TGM"
    # start_date = "2017-06-19"
    # end_date = "2017-06-20"

    # # T31TGK 2023-01-12 is small
    # tile="T31TGK",
    # start_date="2023-01-12",
    # end_date="2023-01-13",
    
    # should download duplicates
    downloaded, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=cloud_min,
        level="LEVEL2A",
        bands=bands,
        dry_run=False)
    
    assert len(downloaded) == 1

    # # simulate old version
    # for f in unzip_files:
    #     if f.exists():
    #         new_file = re.sub("V[0-9]-[0-9]$", "V1-1", f)
    #         f.move(new_file)

    # it should download duplicates (again)
    downloaded, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=cloud_max,
        level="LEVEL2A",
        bands=bands,
        dry_run=False)
    
    assert len(downloaded) == 2
    assert len(unzip_files) == 2
    assert sum([f.exists() for f in unzip_files])==1
    # check if one of the files is considered as merged
    df = get_local_maja_files(unzip_dir)
    assert len(df.merged_id.drop_duplicates()) == 1
    for f in unzip_files:
        if f.exists():
            assert (f / "MASKS").is_dir()
            assert len((f / "MASKS").glob("*CLM_R*.tif"))==2

    # nothing should be downloaded
    zip_files, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=cloud_max,
        level="LEVEL2A",
        bands=bands,
        dry_run=False)
    
    assert len(zip_files) == 0
    assert len(unzip_files) == 0


    # nothing should be downloaded
    zip_files, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=cloud_min,
        level="LEVEL2A",
        bands=bands,
        dry_run=False)
    
    assert len(zip_files) == 0
    assert len(unzip_files) == 0


def test_patch_merged_scenes():
    from path import Path
    from fordead.theia_preprocess import patch_merged_scenes
    unzip_base_dir = Path("/media/boissieu/sshfs_fordead/Data/SENTINEL")
    zip_base_dir = Path("/media/boissieu/sshfs_fordead/Data/SENTINEL/Zipped")
    for unzip_dir in unzip_base_dir.glob("T*"):
        zip_dir = zip_base_dir/unzip_dir.name
        patch_merged_scenes(zip_dir, unzip_dir, dry_run=True)