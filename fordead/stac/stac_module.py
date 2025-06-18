# author(s) : Kenji Ose / Raphael Dutrieux / Florian de Boissieu

from datetime import datetime
import geopandas as gp
import json
import numpy as np
import pandas as pd
from pathlib import Path
import planetary_computer

import pystac
import pystac_client
import re
from shapely.geometry import box, Polygon
from shapely import to_geojson

import dinamis_sdk 
from fordead.stac.theia_collection import build_theia_collection, ItemCollection, bbox_to_wgs, get_rio_info
from fordead.import_data import TileInfo
import fordead.stac.stac_module as st



def get_vectorBbox(vector_path):
    """
    get layer bounding box

    - vector_path: <string> vectorfile path

    return: <list> of bbox coordinates in lat/lon 
            [lon_min, lat_min, lon_max, lat_max]
    """
    obs = gp.read_file(vector_path)
    obs = obs.to_crs("EPSG:4326")
    bbox = list(obs.total_bounds)
    return bbox

def get_bbox(vector):
    """
    get layer bounding box

    - vector: <geopanda dataframe> vector

    return: <list> of bbox coordinates in lat/lon 
            [lon_min, lat_min, lon_max, lat_max]
    """
    vector = vector.to_crs("EPSG:4326")
    bbox = list(vector.total_bounds)
    return bbox

PLANETARY = dict(
    url = "https://planetarycomputer.microsoft.com/api/stac/v1",
    collection = "sentinel-2-l2a",
    cloud_cover = "eo:cloud_cover",
    mgrs_tile = "s2:mgrs_tile"
)

THEIASTAC = dict(
    url = 'https://stacapi-cdos.apps.okd.crocc.meso.umontpellier.fr',
    collection = "sentinel2-l2a-theia",
    cloud_cover = "eo:cloud_cover",
    mgrs_tile = "s2:mgrs_tile"
)

def getItemCollection(start_date, end_date, bbox, cloud_nb = 100, source = PLANETARY):
    """
    get item collection from Theia STAC or Microsoft Planetary Computer

    - start_date: <string> start date yyyy-mm-dd
    - end_date: <string> end date yyyy-mm-dd
    - bbox: <list> bounding box [lon_min, lat_min, lon_max, lat_max]
    - cloud_nb: <int> cloud percentage [default: 0]

    return pystac itemcollection object
    """
    catalog = pystac_client.Client.open(source["url"])
    time_range = f"{start_date}/{end_date}" #"2021-07-01/2021-12-01"
    bbox = bbox
    cloud_nb = cloud_nb

    # recherche avec filtre de type 'query'
    search = catalog.search(
        collections=source["collection"],
        bbox=bbox,
        datetime=time_range,
        query={source["cloud_cover"]: {"lt": float(cloud_nb)}},
        sortby="datetime",
        # seems to solve the duplicates issue if less than 1000 items
        # see https://github.com/microsoft/PlanetaryComputer/issues/163
        limit=1000
    )

    # itemsColl = search.pages()
    itemsColl = search.item_collection()
    # Waiting for standardization of THEIASTAC "sentinel-2-l2a-theia" properties:
    # until then, uniformize used properties for the following
    if source["cloud_cover"] != PLANETARY["cloud_cover"]:
        cloud_cover = source["cloud_cover"]
        for item in itemsColl:
            item.properties[PLANETARY["cloud_cover"]] = item.properties[cloud_cover]

    if source["mgrs_tile"] != PLANETARY["mgrs_tile"]:
        mgrs_tile = source["mgrs_tile"]
        for item in itemsColl:
            item.properties[PLANETARY["mgrs_tile"]] = re.sub(
                "^T", "", item.properties[mgrs_tile])

    return ItemCollection(itemsColl)

# Conversion de tous les items découverts par la recherche, 
# on crée un objet ItemcCollection pour avoir la liste des items. 
# from sys import sys.exit
def get_s2_epsgXY(item_collection):
    """
    Using an item collection, returns a list with all tilenames, the epsg of the corresponding sentinel-2 data, as well as the list of longitude and latitude coordinates of the tile in epsg 4326.
    """
    
        
    corner_names = []
    x_coordinates = []
    y_coordinates = []
    s2_mgrs_tiles = []
    epsg_list = []
    for item in item_collection:
        # print(item.properties["eo:cloud_cover"])
        # Get properties from the item
        s2_mgrs_tile = "T" + item.properties.get('s2:mgrs_tile')
        if "proj:epsg" in item.properties:
            epsg = item.properties["proj:epsg"]
        else:
            epsg = set([asset.extra_fields["proj:epsg"] for asset in item.assets.values() if "proj:epsg" in asset.extra_fields])
            if len(epsg) > 1:
                raise Exception("Multiple proj:epsg found in item nor assets: "+ item.id)
            elif len(epsg) == 1:
                epsg = list(epsg)[0]
            else:
                # looking in assets for proj:wkt2
                wkt2 = set([asset.extra_fields["proj:wkt2"] for asset in item.assets.values() if "proj:wkt2" in asset.extra_fields])
                if len(wkt2) > 1:
                    raise Exception("Multiple proj:wkt2 found in item nor assets: "+ item.id)
                if len(wkt2) == 0:
                    raise Exception("No proj:epsg or proj:wkt2 found in item nor assets: "+ item.id)
                from rasterio.crs import CRS
                epsg = CRS.from_wkt(list(wkt2)[0]).to_epsg()
        # print(epsg)
        # Extract the coordinates
        coordinates = item.geometry['coordinates'][0]
        
        for i, (x, y) in enumerate(coordinates):
            if len(coordinates) == 5:
                corner_name = f'Corner{i+1}'  # Create corner names (Corner1, Corner2, etc.)
                corner_names.append(corner_name)
                x_coordinates.append(x)
                y_coordinates.append(y)
                s2_mgrs_tiles.append(s2_mgrs_tile)
                epsg_list.append(epsg)
    
    # Create a Pandas DataFrame
    data = {
        'tilename': s2_mgrs_tiles,
        'epsg' : epsg_list,
        'Corner': corner_names,
        'x': x_coordinates,
        'y': y_coordinates
    }
    df = pd.DataFrame(data)
    
    # Calculate the min and max for each corner of each tile
    grouped = df.groupby(['tilename', 'Corner',"epsg"])
    min_max = grouped.agg({'x': ['min', 'max'], 'y': ['min', 'max']})
    min_max = grouped.agg({'x': ['min', 'max'], 'y': ['min', 'max']})

    min_max.reset_index(inplace=True)
    min_max.columns = ['tilename', 'Corner', "epsg",'xmin', 'xmax', 'ymin', 'ymax']
    
    out_list = []
    for tile in np.unique(min_max.tilename):
        # print(tile)
        # for corner in ["Corner1","Corner2","Corner3","Corner4","Corner5"]:
        corner1 = min_max[(min_max["tilename"] == tile) & (min_max["Corner"] == "Corner1")]
        corner2 = min_max[(min_max["tilename"] == tile) & (min_max["Corner"] == "Corner2")]
        corner3 = min_max[(min_max["tilename"] == tile) & (min_max["Corner"] == "Corner3")]
        corner4 = min_max[(min_max["tilename"] == tile) & (min_max["Corner"] == "Corner4")]
        # corner5 = min_max[(min_max["tilename"] == tile) & (min_max["Corner"] == "Corner5")]
        
        lon_point_list = [corner1.xmin.iloc[0],corner2.xmin.iloc[0],corner3.xmax.iloc[0],corner4.xmax.iloc[0]]
        lat_point_list = [corner1.ymax.iloc[0],corner2.ymin.iloc[0],corner3.ymin.iloc[0],corner4.ymax.iloc[0]]
        tileinfo = [tile, corner1.epsg.iloc[0], lon_point_list, lat_point_list]
        out_list.append(tileinfo)
    return out_list


def tile_filter(item_collection, tile_name):
    """
    Filter S2 item_collection according to Tile ID

    - item_collection: <pystac object> input S2 item_collection
    - tile_name: <string> S2 tile ID 

    return: <pystac object> filtered item_collection
    """
    filt_item = []
    for item in item_collection:
        if item.properties["s2:mgrs_tile"] == tile_name[1:]:
            filt_item.append(item)
    new_coll = pystac.item_collection.ItemCollection(filt_item)
    return new_coll



def get_items_tiles(item_collection):
    """
    Link S2 tile with corresponding EPSG

    - item_collection: <pystac object> S2 item collection

    return: <list> of list with following elements [tile name, epsg code]
    """
    out_list = []
    for item in item_collection:
        tile = item.properties['s2:mgrs_tile']
        epsg = item.properties["proj:epsg"]
        info = [tile, epsg]
        if info not in out_list:
            out_list.append(info)
    return out_list



    
S2_THEIA_BANDS = [f"B{i+1}" for i in range(12)]+["B8A"]
S2_SEN2COR_BANDS = [f"B{i+1:02}" for i in range(12)]+["B8A"]
def get_harmonized_planetary_collection(start_date, end_date, obs_bbox, lim_perc_cloud, tile=None, sign=False):
    """
    Get the planetary item collection with band names
    and offset harmonized with THEIA format.

    Parameters
    ----------
    start_date : str
        Start of the period.
    end_date : str
        End of the period.
    obs_bbox : list
        bounding box [lon_min, lat_min, lon_max, lat_max]
    lim_perc_cloud : float
        Max cloud cover between 0 et 1.
    tile : str, optional
        The name of a single tile used to filter the collection. If None, all tiles are kept in the collection. The default is None.
    sign : bool
        Should the collection be signed.
    Returns
    -------
    fordead.stac.stac_module.ItemCollection
        The item collection.
    
    Notes
    -----
    The offset is harmonized with `harmonize_sen2cor_offset`.
    """
    
    corresp_keys = {'B01' : 'B1', 'B02' : 'B2', 'B03' : "B3", 'B04' : "B4", 'B05' : "B5", 
                    'B06' : "B6", 'B07' : "B7", 'B08' : "B8", 'B09' : "B9", 'SCL' : "Mask"}
    
    collection = getItemCollection(start_date, end_date, obs_bbox, int(lim_perc_cloud*100), source=PLANETARY)
    # item_list = []
    for item in collection:
        # item.properties["cloud_cover"] = item.properties['eo:cloud_cover']
        item.properties["tilename"] = "T" + item.properties['s2:mgrs_tile']
        #Changing B01 to B1, up to B9, and SCL to Mask in item asset names

        for key_change in corresp_keys:
            if key_change in item.assets:
                item.assets[corresp_keys[key_change]] = item.assets.pop(key_change)
            
    # for item in collection:
    #     print(item.properties["cloud_cover"])
    if tile is not None:
        print("Extracting Sentinel-2 data from "+ tile + " collected from Microsoft Planetary Computer")
        collection = collection.filter(filter=f"tilename = '{tile}'")
    
    harmonize_sen2cor_offset(collection, bands=S2_THEIA_BANDS, inplace=True)
    collection.drop_duplicates(inplace=True)

    if sign:
        collection = planetary_computer.sign_item_collection(collection)
    return collection

def get_harmonized_theiastac_collection(start_date, end_date, obs_bbox, lim_perc_cloud, tile=None, sign=False):
    """
    Get the planetary item collection with band names
    and offset harmonized with THEIA format.

    Parameters
    ----------
    start_date : str
        Start of the period.
    end_date : str
        End of the period.
    obs_bbox : list
        bounding box [lon_min, lat_min, lon_max, lat_max]
    lim_perc_cloud : float
        Max cloud cover between 0 et 1.
    tile : str, optional
        The name of a single tile used to filter the collection. If None, all tiles are kept in the collection. The default is None.
    sign : bool
        Should the collection be signed.
    Returns
    -------
    fordead.stac.stac_module.ItemCollection
        The item collection.
    
    Notes
    -----
    The offset is harmonized with `harmonize_sen2cor_offset`.
    """
    
    corresp_keys = {'B01' : 'B1', 'B02' : 'B2', 'B03' : "B3", 'B04' : "B4", 'B05' : "B5", 
                    'B06' : "B6", 'B07' : "B7", 'B08' : "B8", 'B09' : "B9", 'CLM_R2' : "Mask"}
    
    collection = getItemCollection(start_date, end_date, obs_bbox, int(lim_perc_cloud*100), source=THEIASTAC)
    # item_list = []
    for item in collection:
        # item.properties["cloud_cover"] = item.properties['eo:cloud_cover']
        item.properties["tilename"] = "T" + item.properties['s2:mgrs_tile']
        #Changing B01 to B1, up to B9, and SCL to Mask in item asset names

        for key_change in corresp_keys:
            if key_change in item.assets:
                item.assets[corresp_keys[key_change]] = item.assets.pop(key_change)
            
    # for item in collection:
    #     print(item.properties["cloud_cover"])
    if tile is not None:
        print("Extracting Sentinel-2 data from "+ tile + " collected from theiastac")
        collection = collection.filter(filter=f"tilename = '{tile}'")

        # issue: proj:epsg not all at the same level --> produces NaN in dataframe (inside stac_static) and thus converts to float
        # should be removed when collection is homogeneous...
        harmonize_epsg(collection, inplace=True)

    # harmonize_sen2cor_offset(collection, bands=S2_THEIA_BANDS, inplace=True)
    collection.drop_duplicates(inplace=True)

    if sign:
        collection = dinamis_sdk.sign_item_collection(collection)
    return collection

def harmonize_sen2cor_offset(collection, bands=set(S2_THEIA_BANDS + S2_SEN2COR_BANDS), inplace=False):
    if not inplace:
        collection = collection.copy()
    for item in collection:
        for asset in bands:
            if asset in item.assets:
                if item.properties["datetime"] >= "2022-01-25":
                    item.assets[asset].extra_fields["raster:bands"] = [dict(offset=-1000)]
                else:
                    item.assets[asset].extra_fields["raster:bands"] = [dict(offset=0)]
    if not inplace:
        return collection

def harmonize_epsg(collection, inplace=False):
    # Harmonize epsg at the item level
    # This is a patch for theiastac collection which has
    # the 'proj:epsg' attribute at either at the item level
    # or at the asset level or even no 'proj:epsg' but a 'proj:wkt2'
    #
    # 'collection' expects actually an ItemCollection
    if not inplace:
        collection = collection.copy()
    for item in collection:
        if "proj:epsg" in item.properties and not isinstance(item.properties["proj:epsg"], int):
            if np.isnan(item.properties["proj:epsg"]):
                # looking in assets for proj:epsg
                epsg = set([asset.extra_fields["proj:epsg"] for asset in item.assets.values() if "proj:epsg" in asset.extra_fields])
                if len(epsg) > 1:
                    raise Exception("Multiple proj:epsg found in item nor assets: "+ item.id)
                elif len(epsg) == 1:
                    epsg = list(epsg)[0]
                else:
                    # looking in assets for proj:wkt2
                    wkt2 = set([asset.extra_fields["proj:wkt2"] for asset in item.assets.values() if "proj:wkt2" in asset.extra_fields])
                    if len(wkt2) > 1:
                        raise Exception("Multiple proj:wkt2 found in item nor assets: "+ item.id)
                    if len(wkt2) == 0:
                        raise Exception("No proj:epsg or proj:wkt2 found in item nor assets: "+ item.id)
                    from rasterio.crs import CRS
                    epsg = CRS.from_wkt(list(wkt2)[0]).to_epsg()
                item.properties["proj:epsg"] = int(epsg)
            else:
                item.properties["proj:epsg"] = int(item.properties["proj:epsg"])
    if not inplace:
        return collection

def get_harmonized_theia_collection(sentinel_source, tile_cloudiness, start_date, end_date, lim_perc_cloud, tile):
    # lim_cloud_cover = int(lim_perc_cloud*100)
    corresp_keys = {'CLM' : "Mask"}
    
    sentinel_source = Path(sentinel_source)
    sen_dir_path =  sentinel_source / tile
    
    if not sentinel_source.is_dir():
        raise Exception("Unrecognized sentinel_source")
    
    if not sen_dir_path.is_dir():
        raise Exception("No directory found at " + str(sen_dir_path))

    print("Extracting Sentinel-2 data from " + tile + " stored locally")

    collection = build_theia_collection(sen_dir_path)
    
    
    for item in collection:
        
        #Adding cloud_cover to theia_collection 
        if tile_cloudiness is not None:
            try:
                date_datetime = datetime.strptime(item.properties["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                date_datetime = datetime.strptime(item.properties["datetime"], "%Y-%m-%dT%H:%M:%SZ")
            try:
                acqu_cloudiness = tile_cloudiness[tile_cloudiness.Date == date_datetime.strftime("%Y-%m-%d")]["cloudiness"].values[0]
            except Exception as e:
                # traceback_str = traceback.format_exc()
                print(f"Error: {e}\nCloudiness does not seem to have been extracted for this tile")
            item.properties["cloud_cover"] = acqu_cloudiness
        
        #Changing 'CLM' to 'Mask' 
        for key_change in corresp_keys:
            item.assets[corresp_keys[key_change]] = item.assets.pop(key_change) #Might be changed in the future

    collection = collection.filter(datetime=f"{start_date}/{end_date}")
    
    if tile_cloudiness is not None : collection = collection.filter(filter = f'cloud_cover < {lim_perc_cloud}') 
    
    
    if len(collection) == 0:
        print("Collection is empty")        
    
    return collection
    
def get_tile_collection(tile, include_s2=False):
    if not isinstance(tile, TileInfo) and Path(tile).is_dir():
        tile = TileInfo(tile)
    tile = tile.import_info()
    # update to latest content of results directories...
    tile.getdict_datepaths(key = "VegetationIndex", path_dir = tile.paths["VegetationIndexDir"])
    tile.getdict_datepaths(key = "Masks", path_dir = tile.paths["MaskDir"])
    tile.getdict_datepaths(key = "Anomalies", path_dir = tile.paths["AnomaliesDir"])
    items = []
    for date in tile.paths["VegetationIndex"]:
        bbox = list(tile.raster_meta["extent"])
        epsg =  tile.raster_meta["crs"].to_epsg()
        geometry = json.loads(to_geojson(box(*bbox)))
        shape = tile.raster_meta["shape"]
        transform = list(tile.raster_meta["transform"])
        
        bbox_wgs, geom_wgs, centroid = bbox_to_wgs(bbox, epsg)
        assets = {}
        for key in ["VegetationIndex","Masks","Anomalies"]:
            if date in tile.paths[key]:
                href = str(tile.paths[key][date])
                p = {
                        "href": href,
                        "type": pystac.MediaType.NETCDF if href.endswith(".nc") else pystac.MediaType.GEOTIFF,
                        "roles": ["data"],
                        "proj:epsg": epsg,
                        "proj:bbox": bbox,
                        "proj:geometry": geometry,
                        "proj:shape": shape,
                        "proj:transform": transform,
                        "gsd": tile.raster_meta["transform"][0],
                    }
                # if key == "VegetationIndex":
                #     p["raster:bands"] = [dict(offset=0, scale=1, nodata=-1, data_type="int16")]
                # else:
                #     p["raster:bands"] = [dict(offset=0, scale=1, nodata=0, data_type="int8")]
                assets[key] = pystac.Asset.from_dict(p)
        if include_s2:
            key = "Sentinel"
            if date in tile.paths[key]:
                for band in tile.paths[key][date]:
                    href = str(tile.paths[key][date][band])
                    bbox1, media_type, gsd, meta = get_rio_info(href)
                    
                    if bbox != list(bbox1):
                        raise Exception("Sentinel assets should have the same bbox")
                    assets[band] = pystac.Asset.from_dict(
                        {
                            "href": href,
                            "type": media_type,
                            "roles": ["data"],
                            "proj:epsg": epsg,
                            "proj:bbox": bbox,
                            "proj:geometry": geometry,
                            "proj:shape": (meta["height"], meta["width"]),
                            "proj:transform": list(meta["transform"]),
                            "gsd": gsd,
                        }
                    )
        item_info = {
            "id": date,
            "datetime": pd.to_datetime(date),
            "geometry": geom_wgs,            
            "bbox": bbox_wgs,
            "properties": {"proj:epsg": epsg,},
            "assets": assets,
            # Add XL - pb with Stac schema
            #"stac_extensions": [pystac.extensions.projection.SCHEMA_URI,
            #                    pystac.extensions.raster.SCHEMA_URI],
            }
        item = pystac.Item(**item_info)
        item.validate()
        items.append(item)
    
    collection = ItemCollection(items)
    return collection

def get_polygons_from_sentinel_planetComp(item_collection, tile_selection = None, outputfile = None):
    """
    get image footprint with specified tile crs
    
    - item_collection: pystac item_collection object

    return: geopandas dataframe
    """    
    
    # for item in item_collection:
    #     print(item.properties)
    
    tilesinfo = st.get_s2_epsgXY(item_collection)
    # tilesinfo = st.coord2epsg(tile_extents_4326, in_epsg = "4326", out_epsg = str(epsg))   
    
    if tile_selection is not None:
        tilesinfo = [tile for tile in tilesinfo if tile[0] in tile_selection]
        
    # i=0
    for i in range(len(tilesinfo)):
        # print(i)
        # tile.xmin
        
        # lon_point_list = [tile[2][0][0],tile[2][1][0],tile[2][1][0],tile[2][0][0]]
        # lon_point_list = [tile.xmin,tile.xmax,tile.xmax,tile.xmin]
        # lat_point_list = [tile.ymin,tile.ymin,tile.ymax,tile.ymax]
        #polygon_geom = gp.points_from_xy(lon_point_list, lat_point_list)
        polygon_geom = Polygon(zip(tilesinfo[i][2], tilesinfo[i][3]))
        polygon = gp.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])
        polygon.insert(1, "area_name", tilesinfo[i][0])
        polygon.insert(2, "epsg", f'{tilesinfo[i][1]}')
        
        if i == 0:
            concat_areas = polygon
        else:
            # if concat_areas.crs != polygon.crs:
            #         polygon = polygon.to_crs(concat_areas.crs)
            concat_areas = pd.concat([concat_areas,polygon])
        i +=1
        
    
    if outputfile != None:
        concat_areas.to_file(outputfile) # a sécuriser
    
    return concat_areas


if __name__ == '__main__':
    obs_path = "D:/PROJETS/PRJ_FORDEAD/TEST_STAC/data/observations_tuto.shp"
    obs_bbox = get_vectorBbox(obs_path)
    print(obs_bbox)

    coll = getItemCollection("2021-07-01", "2021-12-01", obs_bbox, 20)
    print(coll.items)
    
