
from fordead.theia_preprocess import maja_download

def test_download(output_dir):
    zip_dir = (output_dir / "download" / "zip").rmtree_p().makedirs_p()
    unzip_dir = (output_dir / "download" / "unzip").rmtree_p().makedirs_p()
    # 31TGM 2018-08-11 is duplicate with cloud_cover [41,52]
    tile = "T31TGM"
    start_date = "2018-08-11"
    end_date = "2018-08-12"

    # # 31TGM 2017-06-19 is duplicate with cloud_cover [3, 4]
    # tile = "T31TGM"
    # start_date = "2017-06-19"
    # end_date = "2017-06-20"

    # # T31TGK 2023-01-12 is small
    # tile="T31TGK",
    # start_date="2023-01-12",
    # end_date="2023-01-13",
    
    # should download duplicates
    zip_files, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=50,
        level="LEVEL2A",
        dry_run=False)
    
    assert len(zip_files) == 1

    # # simulate old version
    # for f in unzip_files:
    #     if f.exists():
    #         new_file = re.sub("V[0-9]-[0-9]$", "V1-1", f)
    #         f.move(new_file)

    # it should download duplicates (again)
    zip_files, unzip_files = maja_download(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        zip_dir=zip_dir,
        unzip_dir=unzip_dir,
        lim_perc_cloud=60,
        level="LEVEL2A",
        dry_run=False)
    
    assert len(zip_files) == 2
    assert sum([f.exists() for f in unzip_files])==1
