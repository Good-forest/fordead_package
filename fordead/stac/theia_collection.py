""""Notes:
    A STAC collection or a catalog is a nested dict (json) listing a number of items
    (e.g. a remote sensing acquisition) and their assets (e.g. bands)

    A tutorial on Landsat, but transposable for Sentinel or others is available here:
    https://pystac.readthedocs.io/en/stable/tutorials/creating-a-landsat-stac.html
    
    # Items

    The required fields for an item (without type and stac_version that are automatically filled by pystac):
        - id: name underwhich the item will be registered
        - geometry: a geojson geometry in dict, see shapely.to_geojson followed by json.loads
        - bbox (List[float]): the bounding box of the item
        - properties (dict): with at least field "datetime" for the date of acquisition.
        - stac_extensions (List[str]): list of the schema URIs used for validation,
        e.g. pystac.extensions.eo.SCHEMA_URI

    # Assets

    The required fields for a stac asset is the `href` (i.e. the path to the file, e.g. a raster file) and
    the `key` (usually the band name) under which it will be registered in the Item.

    Optional but recommended:
        - title (str): usually the band name
        - type (str): see `list(pystac.MediaType)` or
        https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#working-with-media-types
        - roles (List[str]): thumbnail, overview, data, metadata
        - extra_fields (dict): all the rest of the metadata, in particular exetnsion fields

    Other fields are also recommended, especially for satellite data:
    https://github.com/radiantearth/stac-spec/blob/master/item-spec/common-metadata.md
    and https://github.com/radiantearth/stac-spec/blob/master/item-spec/common-metadata.md#instrument

    With the Electro-Optical extension the additionnal fields (extra_fields) are
        - eo:bands (List[dict]): each item of the list corresponds to a band of the band file
        see https://github.com/stac-extensions/eo/#band-object for specs (names and types)
            - name (str): band name
            - common_name (str): the color name for the reflectance bands
            - description (str)
            - center_wavelength (number)
            - full_width_half_max (number)
            - solar_illumination (number)
    
    With the Projection extension the additionnal fields are:
    See https://github.com/stac-extensions/projection/#projection-extension-specification
    and https://github.com/stac-extensions/projection/#best-practices
        - proj:epsg
        - proj:wkt2
        - proj:projjson
        - proj:geometry
        - proj:bbox
        - proj:centroid (dict(lat, lon))
        - proj:shape (List[height, width]): https://github.com/stac-extensions/projection/#projshape
        - proj:transform
    The projection information can be set at the item level if it is the same for
    all bands. Recommendations are to set it at the asset level if it differs from
    
    Although the extension fields are all "optional", if used it must be in the good format,
    the item validation fails otherwise.

    Other extensions are available although not used here,
    see https://github.com/radiantearth/stac-spec/tree/master/extensions#stable-stac-extensions and
    https://stac-extensions.github.io/
"""


import datetime
import json
import pystac
import geopandas as gpd
from pystac.item_collection import ItemLike
import rasterio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.vrt import WarpedVRT
from path import Path
import re
from pandas import DataFrame, to_datetime
import pandas as pd
from pystac.media_type import MediaType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pystac.extensions import eo, projection
import stac_static

from shapely.geometry import box
from shapely import to_geojson
import warnings
import stackstac
from stac_static.search import to_geodataframe
from stac_geoparquet import __version__ as sg_version

# Adds GDAL_HTTP_MAX_RETRY and GDAL_HTTP_RETRY_DELAY to
# stackstac.rio_reader.DEFAULT_GDAL_ENV
# https://github.com/microsoft/PlanetaryComputerExamples/issues/279
# while waiting for a PR to be merged: https://github.com/gjoseph92/stackstac/pull/232
# See also https://github.com/gjoseph92/stackstac/issues/18
DEFAULT_GDAL_ENV = stackstac.rio_reader.LayeredEnv(
    always=dict(
        GDAL_HTTP_MULTIRANGE="YES",  # unclear if this actually works
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        # ^ unclear if this works either. won't do much when our dask chunks are aligned to the dataset's chunks.
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
    ),
    open=dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        # ^ stop GDAL from requesting `.aux` and `.msk` files from the bucket (speeds up `open` time a lot)
        VSI_CACHE=True
        # ^ cache HTTP requests for opening datasets. This is critical for `ThreadLocalRioDataset`,
        # which re-opens the same URL many times---having the request cached makes subsequent `open`s
        # in different threads snappy.
    ),
    read=dict(
        VSI_CACHE=False,
        # ^ *don't* cache HTTP requests for actual data. We don't expect to re-request data,
        # so this would just blow out the HTTP cache that we rely on to make repeated `open`s fast
        # (see above)
        # GDAL_CACHEMAX="500MB"
    ),

)

#### Generic functions and classes ####
def valid_name(x, pattern, directory=False):
   x = Path(x)
   reg = re.compile(pattern)
   if len(reg.findall(x.name))==1 and (x.is_dir() == directory):
      return True
   else:
      return False

def get_rio_info(band_file):
    # band_file = "~/git/fordeadv2/fordead_data-main/sentinel_data/dieback_detection_tutorial/study_area/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1_FRE_B11.tif"
    band_file = Path(band_file).expand()
    with rasterio.open(band_file) as src:
        bbox = src.bounds
        meta = src.meta
        media_type = get_media_type(src)
        gsd = src.res[0]
        
    return bbox, media_type, gsd, meta

def get_media_type(
    src_dst: Union[DatasetReader, DatasetWriter, WarpedVRT, MemoryFile]
) -> Optional[pystac.MediaType]:
    """Find MediaType for a raster dataset.
    Copied from rio-stac (https://github.com/developmentseed/rio-stac)
    """
    driver = src_dst.driver

    if driver == "GTiff":
        if src_dst.crs:
            return pystac.MediaType.GEOTIFF
        else:
            return pystac.MediaType.TIFF

    elif driver in [
        "JP2ECW",
        "JP2KAK",
        "JP2LURA",
        "JP2MrSID",
        "JP2OpenJPEG",
        "JPEG2000",
    ]:
        return pystac.MediaType.JPEG2000

    elif driver in ["HDF4", "HDF4Image"]:
        return pystac.MediaType.HDF

    elif driver in ["HDF5", "HDF5Image"]:
        return pystac.MediaType.HDF5

    elif driver == "JPEG":
        return pystac.MediaType.JPEG

    elif driver == "PNG":
        return pystac.MediaType.PNG

    warnings.warn("Could not determine the media type from GDAL driver.")
    return None

def stac_proj_info(bbox, gsd, meta):
    """Projection information returned in the STAC format.

    It converts typical 

    Parameters
    ----------
    bbox : Bounds or list object
        Bounding box, e.g. rasterio `bounds` attribute.
    gsd : float
        Ground sample distance, e.g. rasterio `res[0]` attribute.
    meta : dict
        Metadata dict returned by rasterio.

    Returns
    -------
    dict
        epsg, wkt2, geometry, centroid, bbox, shape, transform
        with prefix `proj:`, and gsd
    """
    epsg = meta["crs"].to_epsg()
    _, _, centroid = bbox_to_wgs(bbox, meta["crs"])
    proj = dict(
        epsg = epsg,
        wkt2 = meta["crs"].to_wkt(),
        # geometry = bbox_to_geom(bbox),
        geometry = json.loads(to_geojson(box(*bbox))),
        centroid = centroid,
        bbox = list(bbox),
        shape = (meta["height"], meta["width"]),
        transform = list(meta["transform"])
    )

    proj_info = {
            f"proj:{name}": value
            for name, value in proj.items()
    }
    proj_info.update(gsd = gsd)
    
    return proj_info

def bbox_to_wgs(bbox, epsg):
    g = gpd.GeoSeries(box(*bbox), crs=epsg)
    g_wgs = g.to_crs(4326)
    bbox = [float(f) for f in g_wgs.total_bounds]
    geom = json.loads(to_geojson(g_wgs.geometry[0]))
    centroid = g.geometry.centroid.to_crs(4326).iat[0]
    centroid = {"lat": float(centroid.y), "lon": float(centroid.x)}
    return bbox, geom, centroid

class ExtendPystacClasses:
    """Add capacities to_xarray and filter to pystac Catalog, Collection, ItemCollection"""

    def to_xarray(self, xy_coords='center', **kwargs): # gdal_env=DEFAULT_GDAL_ENV
        """Returns a DASK xarray()
        
        This is a proxy to stackstac.stac

        Arguments are:
        assets=frozenset({'image/jp2', 'image/tiff', 'image/vnd.stac.geotiff', 'image/x.geotiff'}),
        epsg=None, resolution=None, bounds=None, bounds_latlon=None,
        snap_bounds=True, resampling=Resampling.nearest, chunksize=1024,
        dtype=dtype('float64'), fill_value=nan, rescale=True,
        sortby_date='asc', xy_coords='topleft', properties=True,
        band_coords=True, gdal_env=None,
        errors_as_nodata=(RasterioIOError('HTTP response code: 404'), ),
        reader=<class 'stackstac.rio_reader.AutoParallelRioReader'>

        For details, see [stackstac.stac](https://stackstac.readthedocs.io/en/latest/api/main/stackstac.stack.html)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            arr = stackstac.stack(self, xy_coords=xy_coords, **kwargs) # gdal_env=gdal_env,
        return arr
    
    def filter(self, asset_names=None, **kwargs):
        """Filter items with stac-static search.
        
        Documentation copied from stac-static.

        All parameters correspond to query parameters described in the `STAC API - Item Search: Query Parameters Table
        <https://github.com/radiantearth/stac-api-spec/tree/master/item-search#query-parameter-table>`__
        docs. Please refer to those docs for details on how these parameters filter search results.

        Args:
            ids: List of one or more Item ids to filter on.
            collections: List of one or more Collection IDs or :class:`pystac.Collection`
                instances. Only Items in one
                of the provided Collections will be searched
            bbox: A list, tuple, or iterator representing a bounding box of 2D
                or 3D coordinates. Results will be filtered
                to only those intersecting the bounding box.
            intersects: A string or dictionary representing a GeoJSON geometry, or
                an object that implements a
                ``__geo_interface__`` property, as supported by several libraries
                including Shapely, ArcPy, PySAL, and
                geojson. Results filtered to only those intersecting the geometry.
            datetime: Either a single datetime or datetime range used to filter results.
                You may express a single datetime using a :class:`datetime.datetime`
                instance, a `RFC 3339-compliant <https://tools.ietf.org/html/rfc3339>`__
                timestamp, or a simple date string (see below). Instances of
                :class:`datetime.datetime` may be either
                timezone aware or unaware. Timezone aware instances will be converted to
                a UTC timestamp before being passed
                to the endpoint. Timezone unaware instances are assumed to represent UTC
                timestamps. You may represent a
                datetime range using a ``"/"`` separated string as described in the spec,
                or a list, tuple, or iterator
                of 2 timestamps or datetime instances. For open-ended ranges, use either
                ``".."`` (``'2020-01-01:00:00:00Z/..'``,
                ``['2020-01-01:00:00:00Z', '..']``) or a value of ``None``
                (``['2020-01-01:00:00:00Z', None]``).

                If using a simple date string, the datetime can be specified in
                ``YYYY-mm-dd`` format, optionally truncating
                to ``YYYY-mm`` or just ``YYYY``. Simple date strings will be expanded to
                include the entire time period, for example:

                - ``2017`` expands to ``2017-01-01T00:00:00Z/2017-12-31T23:59:59Z``
                - ``2017-06`` expands to ``2017-06-01T00:00:00Z/2017-06-30T23:59:59Z``
                - ``2017-06-10`` expands to ``2017-06-10T00:00:00Z/2017-06-10T23:59:59Z``

                If used in a range, the end of the range expands to the end of that
                day/month/year, for example:

                - ``2017/2018`` expands to
                ``2017-01-01T00:00:00Z/2018-12-31T23:59:59Z``
                - ``2017-06/2017-07`` expands to
                ``2017-06-01T00:00:00Z/2017-07-31T23:59:59Z``
                - ``2017-06-10/2017-06-11`` expands to
                ``2017-06-10T00:00:00Z/2017-06-11T23:59:59Z``

            filter: JSON of query parameters as per the STAC API `filter` extension
            filter_lang: Language variant used in the filter body. If `filter` is a
                dictionary or not provided, defaults
                to 'cql2-json'. If `filter` is a string, defaults to `cql2-text`.
            
        Notes:
            Argument filter would search into the first level of metadata of the asset.
            If the metadata to filter is a string, it should be used as 
            a string into a string, examples:
             - filter="constellation = 'sentinel-2' and tilename = 'T31UFQ'"
             - filter="tilename in ('T31UFQ', 'T31UFQ')"
            
             In order to filter/select assets, use to_xarray(asset=...) or to_xarray().sel(band=...)
        """
        # issue with proj:epsg that is converted sometimes to float, if 
        # not all of them have it at the same level, e.g. some at item level,
        # others at the asset level...
        res = ItemCollection(stac_static.search(self, **kwargs).item_collection())
        if asset_names is not None:
            for item in res.items:
                item.assets = {k:a for k, a in item.assets.items() if k in asset_names}
        
        return res

    def to_geodataframe(self, **kwargs):
        return to_geodataframe(self)

    def drop_duplicates(self, inplace=False):
        """
        A function to drop duplicates from the collection
        based on the item ids.

        Parameters
        ----------
        inplace : bool, optional
            If True, the collection is modified in place.

        Returns
        -------
        ItemCollection
            The collection with duplicates dropped, if inplace is False.
        
        Notes
        -----
        Duplicates seem to be occuring at search depending on the paging.
        See https://github.com/microsoft/PlanetaryComputer/issues/163
        """
        x=self
        if not inplace:
            x=self.clone()

        index = pd.Series([i.id for i in self]).duplicated()
        if index.any():
            x.items = [i for i, v in zip(x.items, ~index) if v]
        if not inplace:
            return x

class ItemCollection(pystac.ItemCollection, ExtendPystacClasses):
    
    def __init__(self, items: Iterable[ItemLike], **kwargs):
        super().__init__(items, **kwargs)
        if sg_version > "0.3.2":
            remove_uncommon_properties(self, inplace=True)

class Catalog(pystac.Catalog, ExtendPystacClasses):
    pass
#######################################

####
def remove_uncommon_properties(x: ItemCollection, inplace=False):
    if not inplace:
        collection = collection.copy()

    keys = set(x.items[0].properties.keys())
    udiff = []
    for item in x.items:
        keys = keys.intersection(item.properties.keys())
        
    for item in x.items:
        diff = keys.symmetric_difference(item.properties.keys())
        for k in diff:
            item.properties.pop(k)
            udiff.append(k)

    if len(udiff) > 0:
        print(f"Removed properties that were not common to all items:\n{set(udiff)}")

    if not inplace:
        return collection

#### Collection specific fucntions and classes #####
def parse_theia_name(x):
    """
    Parse Sentinel 2 .SAFE directory name and raster file name

    Parameters
    ----------
    x: str
        Theia image directory or Theia band file

    Returns
    -------
    dict
        dictionary of string corresponding to each part of Theia image name or raster file name
    """
    # x = 'fordead_data-main/sentinel_data/dieback_detection_tutorial/study_area/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1'
    # x = 'SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1'

    if isinstance(x, list):
        df = DataFrame([parse_theia_name(f) for f in x])
        sort_names = [n for n in ['date', 'band'] if n in df.keys()]
        return df.sort_values(sort_names)

    props = Path(x).name.stripext().split('_')
    if props[-1] in ["R1", "R2"]:
        pnames = ['sensor', 'datatake', 'level', 'tile', 'D', 'version', 
                'band', "res"]
    else:
        pnames = ['sensor', 'datatake', 'level', 'tile', 'D', 'version', 
                    'FRE', 'band']

    props = {pnames[i]:p for i, p in enumerate(props)}
    props['date'] = to_datetime(props['datatake'], format="%Y%m%d-%H%M%S-%f") # datetime.datetime.strptime(, '%Y%m%d')
    props['path'] = x

    return props

# Expected pattern for a Theia image name
# x = "SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1"
theia_image_pattern = "_".join(
       ["SENTINEL2[AB]",
        "[0-9]{8}-[0-9]{6}-[0-9]{3}",
        "L2A",
        "T[0-9A-Z]{5}",
        "[A-Z]",
        "V[0-9]-[0-9]"])

# Expected pattern for a Theia band file name
# x = "SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1_FRE_B2"
theia_band_pattern = "_".join([
    theia_image_pattern,
    "FRE",
    "B[1-9][0-9A]*.tif$"
    ])

# Expected band names
theia_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
               'B8', 'B8A', 'B11', 'B12', 'CLM']

# Expected band order in the item
theia_bands_index = {v:i for i, v in enumerate(theia_bands)}

# Common names corresponding to band names
theia_band_common_names = dict(
    B2="blue", B3="green", B4="red",
    B5="rededge", B6="rededge", B7="rededge",
    B8="nir", B8A="nir08", B11="swir16", B12="swir22",
)

def build_theia_collection(input_dir, collection_name="my_theia_collection"):
    """Build a Theia collection with the images in input_dir

    As the metadata file *_MTD_ALL.xml was missing,
    the item and assets fields are filled thanks to 
    the image directory name and the rasterio properties.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing the Theia image files.
    
    collection_name : str, optional
        Collection name, by default "my_theia_collection"

    Returns
    -------
    pystac.ItemCollection
        This collection can then be saved into a unique json file
    """
    input_dir = Path(input_dir).expand()
    image_list =  [d for d in input_dir.dirs() if valid_name(d, theia_image_pattern, True)]
    items = []
    for image in image_list:
        items.append(
            MyTheiaImage(image).create_item()
        )
    return ItemCollection(items)    

def theia_stac_asset_info(band_file):
    """Parse theia band information for stac.
    
    It uses the file basename to get the band and 
    the rasterio header info for the rest.
    """
    
    band_file = Path(band_file).expand()
    
    # parse band file to get band name
    props =  parse_theia_name(band_file)
    band_name = props['band']
    
    # get rasterio information
    bbox, media_type, gsd, meta = get_rio_info(band_file)

    # build stac information
    stac_fields = {
        "key": band_name,
        "href": band_file,
        "type" : media_type,
        "instruments": ["MSI"],
        "constellation": "sentinel-2 theia",
    }

    # get roles and eobands depending on 
    if band_name in theia_band_common_names:
        roles = ["reflectance"]
        eoband = {
            "name": band_name,
            "common_name": theia_band_common_names[band_name],
            "description": f"S2 Theia band {band_name}",
        }
        stac_fields.update({"eo:bands":[eoband]}) # list required

    elif band_name=="CLM":
        # Classification extension could be used
        roles = ["cloud"]
    else:
        roles = ["data"]

    stac_fields.update(roles=roles)

    # add projection as the resolution is not the same for all bands
    # It could be set at the item level otherwise.
    proj_info = stac_proj_info(bbox, gsd, meta)
    stac_fields.update(proj_info)
    
    return stac_fields

def theia_stac_item_info(image_dir, assets=None):
    image_dir = Path(image_dir).expand()

    props = parse_theia_name(image_dir)
    datetime = props['date']
    id = '_'.join(image_dir.name.split('_'))
    properties = {
            "title": id,
            "tilename": props["tile"],
            "constellation": "sentinel-2 theia",
    }

    ### common to any other item ###
    geometry = bbox = None
    if assets is not None:
        # The item's bbox and geometry could be computed as the ones
        # englobing all assets. Here it is expected that all assets
        # have the same bbox and geometry the first asset taken as reference.
        _, asset = assets[0]
        bbox, geometry, _ = bbox_to_wgs(
            asset.extra_fields["proj:bbox"], 
            asset.extra_fields["proj:epsg"])

        properties.update({
                "proj:epsg" : asset.extra_fields["proj:epsg"]
            })
    
    # In case of ItemCollection, href is not included as it designates the json file
    # where the item description should be saved. Using pystac Collection or Catalog,
    # the href would be filled when saved.
    stac_fields = dict(
        id = id,
        datetime = datetime,
        properties = properties,
        assets = {k:v for k,v in assets},
        bbox = bbox,
        geometry = geometry,
        stac_extensions = [eo.SCHEMA_URI, projection.SCHEMA_URI]
    )

    return stac_fields

class MyTheiaImage(object):
    
    def __init__(self, image_dir):
        image_dir = Path(image_dir).expand()
        self.image_dir = image_dir

    def get_band_files(self):
        band_files = [f for f in self.image_dir.files() if valid_name(f, theia_band_pattern)]
        clm_masks = (self.image_dir / "MASKS").glob("*CLM_R*.tif")
        if len(clm_masks) == 0:
            raise FileNotFoundError(f"File *CLM_Rx.tif not found in {self.image_dir}")
        if len(clm_masks) > 1:
            raise ValueError(f"Multiple files *CLM_Rx.tif found in {self.image_dir}")
        
        band_files.append(clm_masks[0])
        return band_files

    def create_assets(self):
        asset_list = []
        for band_file in self.get_band_files():
            stac_info = theia_stac_asset_info(band_file)
            key = stac_info.pop("key")
            asset_list.append((key, pystac.Asset.from_dict(stac_info)))
        
        # sort assets in the order of theia_band_index
        df =  DataFrame(asset_list, columns=['band', 'asset'])
        df.sort_values(by='band', key=lambda x: x.map(theia_bands_index), inplace=True)
        df.reset_index(drop=True, inplace=True)
        assets = list(df.itertuples(index=False, name=None))

        return assets
    
    def create_item(self, collection=None, validate=True):

        # create assets
        assets = self.create_assets()
        # prepare item dict
        stac_info = theia_stac_item_info(self.image_dir, assets)
        # create item
        item = pystac.Item(**stac_info)
        # validate item structure        
        if validate:
            item.validate()

        return item

###########################

