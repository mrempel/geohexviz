# imports
import json
import operator
import warnings
from typing import Callable, List, Optional, Dict, Any, Union, Tuple, Iterable
import geopandas as gpd
import pandas as pd
from pyproj import Geod
from geojson import Feature, FeatureCollection
from geopandas import GeoDataFrame, GeoSeries
from h3 import h3
from pandas import DataFrame
from shapely.geometry import Polygon, Point, polygon, LineString, LinearRing, \
    MultiPoint, MultiPolygon, MultiLineString, GeometryCollection
from functools import reduce

GeometryContainer = Union[MultiPolygon, MultiPoint, MultiLineString, GeometryCollection]
AnyGeom = Union[Polygon, Point, LineString, GeometryContainer]

def add_geometry(row) -> Polygon:
    """Returns a Polygon of the hex id in the same row.

    This function returns a Polygon representing the boundaries defined by
    the hex cell in the given row.

    :param row: A row in the dataframe
    :type row:
    :param hex_field_id:
    :type hex_field_id:
    :return: A polygon built from a hex id
    :rtype: Polygon
    """
    points = h3.h3_to_geo_boundary(row[row.index.values[0]], True)
    return Polygon(points)


def _fix_lon(num: float):
    return 360 + num if num < 0 else num


def _fix_lat(num: float):
    return num


def _check_antimeridian(long0, long1):
    return abs(long1 - long0) > 180


def _wraps_lat(poly: Polygon):
    coordinates = list(poly.exterior.coords)
    for x in range(1, len(coordinates)):
        p0 = coordinates[x - 1]
        p1 = coordinates[x]
        if check_crossing(p0[0], p1[0], validate=False):
            return True
    return False


def _fix_polygon_bounds(poly: Polygon):
    wraps = _wraps_lat(poly)
    coordinates = list(poly.exterior.coords)
    new_poly = []
    for cord in coordinates:
        # lon = (360.0 + cord[0]) % 360
        lon = _fix_lon(cord[0]) if wraps else cord[0]
        lat = _fix_lat(cord[1])
        new_poly.append((lon, lat))
    return Polygon(new_poly)


def convert_gdf_crs(gdf: GeoDataFrame, crs: Any = 'EPSG:4326'):
    if gdf.crs:
        return gdf.to_crs(crs)
    else:
        cdf = gdf.copy(deep=True)
        cdf.crs = crs
        return cdf


def check_crossing(lon1: float, lon2: float, validate: bool = True):
    """
    Assuming a minimum travel distance between two provided longitude coordinates,
    checks if the 180th meridian (antimeridian) is crossed.
    """
    if validate and any(abs(x) > 180.0 for x in [lon1, lon2]):
        raise ValueError("longitudes must be in degrees [-180.0, 180.0]")
    return abs(lon2 - lon1) > 180.0


def hexify_dataframe(gdf: GeoDataFrame, hex_resolution: int, add_geom: bool = False, keep_geom: bool = False,
                     old_geom_name: str = None, as_index: bool = True, raise_errors: bool = False) -> GeoDataFrame:
    """Makes a new GeoDataFrame, with the index set as the hex cell ids that each geometry in the geometry column
    corresponds to.

    :param gdf: The GeoDataFrame to place a hex cell id overlay on
    :type gdf: GeoDataFrame
    :param hex_resolution: The resolution of the hexes to be generated
    :type hex_resolution: int
    :param add_geom: Whether to add the hex geometry to the dataframe or not
    :type add_geom: bool
    :param keep_geom: Whether to keep old geometry or not (add_geom=True)
    :type keep_geom: bool
    :param old_geom_name: The name of the column to store the old geometry in (add_geom=True, keep_geom=True)
    :type old_geom_name: str
    :param as_index: Whether to make the hex column the index or not
    :type as_index: bool
    :param raise_errors: Whether to raise errors related to empty geometry or not
    :type raise_errors: bool
    :return: A GeoDataFrame with a hex id index
    :rtype: GeoDataFrame
    """
    if old_geom_name is None:
        old_geom_name = '*OLD GEOMS*'

    cdf = gdf.copy(deep=True)

    def get_multi_shape_hex(shape: GeometryContainer, fn: Callable) -> List[str]:
        """A function to retrieve a list of hex ids for container-like geometries.
        """
        lst = []
        for s in shape:
            lst.extend(fn(s))
        return lst

    def shape_to_hex_ids(shape: AnyGeom) -> List[str]:
        """A semi-recursive function for the retrieving of hex ids.
        """

        if isinstance(shape, Point):
            return [h3.geo_to_h3(lat=shape.y, lng=shape.x, resolution=hex_resolution)]
        elif isinstance(shape, Polygon):
            return list(h3.polyfill(shape.__geo_interface__, hex_resolution, geo_json_conformant=True))
        elif isinstance(shape, LineString) or isinstance(shape, LinearRing):
            return [shape_to_hex_ids(Point(x))[0] for x in shape.coords]
        else:
            return get_multi_shape_hex(shape, shape_to_hex_ids)

    try:
        cdf['HEX'] = cdf.geometry.apply(shape_to_hex_ids)  # astype suppresses warning
    except AttributeError:
        if raise_errors:
            raise ValueError("There was no geometry in the dataframe.")
        return cdf

    # attempt to explode while removing empty cells
    cdf['*HEXLEN*'] = cdf['HEX'].apply(len)
    try:
        cdf = pd.DataFrame.explode(cdf[cdf['*HEXLEN*'] != 0].drop(columns='*HEXLEN*').reset_index(drop=True),
                                   'HEX', ignore_index=True)

        if as_index:
            cdf.set_index('HEX', inplace=True)
    except KeyError:
        if raise_errors:
            raise ValueError("There was no geometry in the dataframe.")
        return cdf

    # add the hex geometries and keep the old ones if specified
    if add_geom:
        if keep_geom:
            cdf[old_geom_name] = cdf.geometry
        cdf = hexify_geometry(cdf.drop(columns='geometry'), hex_col=None if as_index else 'HEX')

    return cdf


def apply_bin_function(hex_gdf: DataFrame, binning_field: str, binning_fn: Callable, binning_args=None,
                       result_name: str = None, **binning_kw):
    """Applies a function to a grouped dataframe (intended for hex use).

    :param hex_gdf: The dataframe to perform the function on
    :type hex_gdf: DataFrame
    :param binning_field: The column within the dataframe to apply the function on
    :type binning_field: str
    :param binning_fn: The function to apply
    :type binning_fn: Callable
    :param binning_args: Arguments for the function
    :type binning_args: Iterable
    :param result_name: The name of the column that contains the result
    :type result_name: str
    :param binning_kw: Keyword arguments for the function
    :type binning_kw: **kwargs
    """
    if binning_args is None:
        binning_args = []
    if result_name is None:
        result_name = '*COLLECTED VALUE*'
    hex_gdf[result_name] = (hex_gdf[binning_field].apply(binning_fn, args=binning_args, **binning_kw))


def bin_by_hexid(hex_gdf: Union[DataFrame, GeoDataFrame], binning_field: str = None, binning_fn: Callable = None,
                 binning_args=None, hex_field: str = None, result_name: str = 'value_field', add_geoms: bool = False,
                 loss_method: bool = True, **binning_kw):
    """Bins a DataFrame by hex cell ids.

    This function assumes the dataframe has a VALID hex and geometry columns.
    Using these columns the data is grouped into what hexagon on the grid they fall into.


    :param hex_gdf: A dataframe representing any kind of hex data
    :type hex_gdf: GeoDataFrame
    :param binning_field: The column that the data will be grouped by
    :type binning_field: str
    :param binning_fn: The function to perform after grouping
    :type binning_fn: Callable
    :param binning_args: Arguments for the binning function
    :type binning_args: *args
    :param hex_field: The location of the hex ids in the dataframe (None->index)
    :type hex_field: str
    :param result_name: The name of the column that contains the grouped result after having the function applied
    :type result_name: str
    :param add_geoms: Whether to add hex geometries after grouping or not
    :type add_geoms: bool
    :param loss_method: Whether or not to use a method that is quicker but provides a loss of data (columns)
    :type loss_method: bool
    :return: A frame containing the binned hex grid and its geometries
    :rtype: GeoDataFrame
    """
    if binning_fn is None:
        binning_fn = lambda lst: len(lst)

    if binning_field is None:
        hex_gdf['binby'] = 1
        binning_field = 'binby'

    hex_gdf.dropna(subset=[binning_field], inplace=True)
    hex_gdf = hex_gdf.groupby(by=hex_field if hex_field is not None else hex_gdf.index)
    # tup = lambda lst: tuple([x for x in lst if np.isnan(x) == False])

    # group by ids aggregate into list
    if loss_method:
        hex_gdf_g = (hex_gdf[binning_field].agg(tuple).to_frame(binning_field))
    else:
        # the no loss method could be sped up (no geometry column)
        hex_gdf_g = (hex_gdf.agg(tuple))

    apply_bin_function(hex_gdf_g, binning_field, binning_fn, binning_args=binning_args,
                       result_name=result_name, **binning_kw)

    if add_geoms:
        hex_gdf_g = hexify_geometry(GeoDataFrame(hex_gdf_g))
        hex_gdf_g.crs = "EPSG:4326"

    return hex_gdf_g


def hexify_geometry(gdf: Union[DataFrame, GeoDataFrame], hex_col: str = None, keep_geoms: bool = False,
                    old_geom_name: str = None) -> GeoDataFrame:
    """Adds the geometry of the hex ids in the given column or index.

    :param gdf: The GeoDataFrame containing the hex ids
    :type gdf: GeoDataFrame
    :param hex_col: The column containing hex ids (None->index)
    :type hex_col: str
    :param keep_geoms: Whether or not to keep old geometry
    :type keep_geoms: bool
    :param old_geom_name: The name for the column containing old geometry (keep_geoms=True)
    :type old_geom_name: str
    :return: A GeoDataFrame with hex ids and their geometries
    :rtype: GeoDataFrame
    """
    if not isinstance(gdf, GeoDataFrame):
        gdf = GeoDataFrame(gdf)

    if old_geom_name is None:
        old_geom_name = '*OLD GEOMS*'

    if keep_geoms:
        gdf[old_geom_name] = gdf.geometry

    if hex_col:
        gdf.geometry = gdf[hex_col].apply(lambda r: Polygon(h3.h3_to_geo_boundary(r, True)))
    else:
        gdf['hexv'] = gdf.index.values
        gdf.geometry = gdf['hexv'].apply(lambda r: Polygon(h3.h3_to_geo_boundary(r, True)))
        gdf.drop(columns='hexv', inplace=True, errors='raise')

    gdf.crs = 'EPSG:4326'
    return gdf


def find_geoms_within_collection(gc, collapse: bool = False) -> set:
    """Finds the geometry types within a collection of geometries.

    :param gc: The collection of geometries
    :param collapse: Whether or not to collapse Multi geometries into their sub counterparts
    :type collapse: bool
    :return: The geometry types found
    :rtype: set
    """
    lst = []

    def helper(g):
        try:
            if collapse or isinstance(g, GeometryCollection):
                [helper(item) for item in g.geoms]
            else:
                lst.append(str(g.geom_type))
        except AttributeError:
            lst.append(str(g.geom_type))

    helper(gc)
    return set(lst)


def get_present_geomtypes(gdf: GeoDataFrame, allow_collections: bool = True, collapse_geoms: bool = False) -> set:
    """Obtains a set of unique geometry types within a geodataframe.

    :param gdf: The geodataframe to find geometry types of
    :type gdf: GeoDataFrame
    :param allow_collections: Whether or not to parse GeometryCollections for their types
    :type allow_collections: bool
    :param collapse_geoms: Whether or not to collapse Multi geometries into their sub geometries
    :type collapse_geoms: bool
    :return: The unique set of geometries within the geodataframe
    :rtype: set
    """
    cdf = gdf.copy(deep=True)
    cdf['*GTYPES*'] = cdf.geom_type

    mask = cdf['*GTYPES*'].isin(['GeometryCollection', 'MultiPoint', 'MultiLineString', 'MultiPolygon'])

    coll = GeoDataFrame(cdf[mask])
    other = GeoDataFrame(cdf[~mask])

    lst = list(other['*GTYPES*'].unique())
    if allow_collections:
        for i, row in coll.iterrows():
            lst.extend(find_geoms_within_collection(row.geometry, collapse=collapse_geoms))
    else:
        lst.append('GeometryCollection')
    return set(lst)


def check_geom_only(gdf: GeoDataFrame, gtype: str, collapse_geoms: bool = False):
    result = get_present_geomtypes(gdf, allow_collections=True, collapse_geoms=collapse_geoms)
    return len(result) == 1 and result.pop() == gtype


def check_geom_in(gdf: GeoDataFrame, gtype: str) -> bool:
    return gtype in gdf.geom_type.values


def unify_geodataframe(gdf: GeoDataFrame) -> GeoDataFrame:
    """Unifies the geometries in a GeoDataFrame into a new GeoDataFrame.

    :param gdf: The input dataframe
    :type gdf: GeoDataFrame
    """

    geom = gdf.unary_union.boundary
    return GeoDataFrame(geometry=[geom], crs='EPSG:4326')


def pointify_geodataframe(gdf: GeoDataFrame, keep_geoms: bool = True, raise_errors: bool = True) -> GeoDataFrame:
    """Makes a new GeoDataFrame, with the geometry all being of Point type, and the index
    representing the original row in the dataframe.

    :param gdf: The GeoDataFrame to convert the geometries of
    :type gdf: GeoDataFrame
    :param keep_geoms: Whether to keep the original geometries in the dataframe or not
    :type keep_geoms: bool
    :param raise_errors: Errors are raised by pandas when the dataframe has no geometry, throw or not
    :type raise_errors: bool
    :return: A GeoDataFrame with a point-only geometry
    :rtype: GeoDataFrame
    """
    cdf = gdf.copy(deep=True)

    def get_multi_shape_points(shape: GeometryContainer) -> List[Point]:
        """A function to retrieve a list of hex ids for container-like geometries.
        """
        lst = []
        for s in shape:
            lst.extend(shape_to_points(s))
        return lst

    def shape_to_points(shape: AnyGeom) -> List[Point]:
        """A semi-recursive function for the retrieving of hex ids.
        """
        if isinstance(shape, Point):
            return [shape]
        elif isinstance(shape, Polygon):
            return [Point(s) for s in shape.exterior.coords]
        elif isinstance(shape, LineString):
            return [shape_to_points(Point(s))[0] for s in shape.coords]
        else:
            return get_multi_shape_points(shape)

    cdf['points'] = cdf.geometry.apply(shape_to_points)

    try:
        cdf['pointslen'] = cdf['points'].apply(len)
        cdf = cdf[cdf['pointslen'] != 0]
        cdf = GeoDataFrame(pd.DataFrame.explode(cdf, 'points'),
                           crs=cdf.crs).drop(columns='pointslen')

        if keep_geoms:
            cdf['old-geometry'] = cdf.geometry
        cdf.geometry = cdf['points']
        return cdf.drop(columns='points', errors='ignore')
    except KeyError:
        if raise_errors:
            raise ValueError("There was an error when converting the dataframe, most likely no geometry present.")
        else:
            return cdf


def convert_dataframe_geometry_to_geodataframe(df: DataFrame, geometry_field: str = 'geometry') -> GeoDataFrame:
    """Converts the given dataframe into a GeoDataFrame based on pre-existing geometry.

    This function converts the given dataframe into a GeoDataFrame based on a pre-existing
    geometry column in the dataframe.

    :param df: Any dataframe with a pre-existing geometry column
    :type df: DataFrame
    :param geometry_field: The column that contains geometry
    :type geometry_field: str
    :return: A geodataframe of the given dataframe
    :rtype: GeoDataFrame
    """

    return GeoDataFrame(df, geometry=df[geometry_field], crs='EPSG:4326')


def conform_geogeometry(gdf: GeoDataFrame, d3_geo: bool = True, fix_polys: bool = True) -> GeoDataFrame:
    """Fixes the winding order and antimeridian crossings for geometries in a GeoDataFrame.

    :param gdf: The geodataframe to conform
    :type gdf: GeoDataFrame
    :param d3_geo: Whether to orient the polygons clockwise or counter-clockwise
    :type d3_geo: bool
    :param fix_polys: Whether to fix antimeridian crossings or not
    :type fix_polys: bool
    :return: The conformed geodataframe
    :rtype: GeoDataFrame
    """

    def conform_geoshape(shape, fn):
        try:
            return fn(shape)
        except AttributeError:
            try:
                lst = []
                for item in shape:
                    try:
                        lst.append(fn(item))
                    except AttributeError:
                        lst.append(item)
                if isinstance(shape, MultiPolygon):
                    return MultiPolygon(lst)
                elif isinstance(shape, GeometryCollection):
                    return GeometryCollection(lst)
                else:
                    return shape
            except (TypeError, ValueError):
                return shape
        except TypeError:
            return shape

    if fix_polys:
        perform = (lambda s: polygon.orient(_fix_polygon_bounds(s), sign=-1)) if d3_geo else (
            lambda s: polygon.orient(_fix_polygon_bounds(s), sign=1))
    else:
        perform = (lambda s: polygon.orient(s, sign=-1)) if d3_geo else (lambda s: polygon.orient(s, sign=1))

    # gdf['valid'] = gdf.is_valid
    geoms = gdf.geometry.apply(conform_geoshape, args=[perform])

    gdf.geometry = list(geoms.values)
    return gdf


def conform_polygon(poly: Union[Polygon, MultiPolygon], rfc7946: bool = True, fix_poly: bool = True) -> Union[
    Polygon, MultiPolygon]:
    """Conforms the given polygon to the given standard.

    :param poly: The polygon to conform
    :type poly: Polygon
    :param rfc7946: The conform standard (True->Clockwise,False->Counterclockwise)
    :type rfc7946: bool
    :param fix_poly: Whether or not to fix the polygons if they cross the anti-meridian (by shifting)
    :type fix_poly: bool
    :return: The conformed Polygon
    :rtype: Polygon
    """
    try:
        if fix_poly:
            poly = _fix_polygon_bounds(poly)
        return polygon.orient(poly, sign=-1) if rfc7946 else polygon.orient(poly, sign=1)
    except AttributeError:
        if fix_poly:
            perform = (lambda p: polygon.orient(_fix_polygon_bounds(p), sign=-1)) if rfc7946 else (
                lambda p: polygon.orient(_fix_polygon_bounds(p), sign=1))
        else:
            perform = (lambda p: polygon.orient(p, sign=-1)) if rfc7946 else (lambda p: polygon.orient(p, sign=1))
        lst = []
        for i in range(len(poly)):
            lst.append(perform(poly[i]))
        return MultiPolygon(lst)


def get_area(poly: Polygon):
    geod = Geod(ellps="WGS84")
    return abs(geod.geometry_area_perimeter(poly)[0])


def repeater_merge(*args, **kwargs):
    return reduce(lambda left, right: pd.merge(left, right, **kwargs), list(args))


def merge_datasets_simple(*args,
                          merge_op: Callable = operator.add,
                          common_columns: Optional[List[str]] = None, keep_columns: Optional[List[str]] = None,
                          drop: bool = True, crs: Optional[str] = None,
                          result_name: Optional[str] = None, errors: str = 'ignore') -> GeoDataFrame:
    """Merges the datasets with the given merge operation.

    :param args: The datasets to merge
    :type args: *args: List[Union[GeoDataFrame, Tuple[str, GeoDataFrame]]]
    :param merge_op: The merge operation to perform on the datasets
    :type merge_op: Callable
    :param keep_columns: Additional columns to keep
    :type keep_columns: List[str]
    :param common_columns: Columns that ALL dataframes share in common
    :type common_columns: List[str]
    :param drop: Whether to drop unnecessary columns or not
    :type drop: bool
    :param crs: The crs of the new dataframe
    :type crs: Optional[str]
    :param result_name: The name of the resulting column within the dataframe
    :type result_name: Optional[str]
    :param errors: The parameter determining errors to be thrown
    :type errors: str
    :return: A merged dataframe
    :rtype: GeoDataFrame
    """
    merged_frame = GeoDataFrame(geometry=[])

    first = args[0]

    if isinstance(first, tuple):
        if not isinstance(first[1], GeoDataFrame):
            raise AttributeError(f"If you input a tuple, the first item must be the datasets name and the second a "
                                 f"GeoDataFrame object. The function got an incorrect tuple: {first}")
        merged_frame.index.set_names(first[1].index.name, inplace=True)
        merged_frame.crs = first[1].crs
    else:
        merged_frame.index.set_names(first.index.name, inplace=True)
        merged_frame.crs = first.crs

    col_name = lambda i: 'vf-' + str(i)
    col_names = []
    ds_num = 1

    common_columns = common_columns if common_columns else []
    keep_columns = keep_columns if keep_columns else []

    for item in args:

        if isinstance(item, tuple):
            if not isinstance(item[1], GeoDataFrame):
                raise AttributeError(f"If you input a tuple, the first item must be the datasets name and the second a "
                                     f"GeoDataFrame object. The function got an incorrect tuple: {item}")
            ds_name, ds_frame = item
        else:
            ds_name, ds_frame = ds_num, item
            ds_num += 1

        if col_name(ds_name) in col_names:
            raise AttributeError("If you put names for datasets, each one should be unique.")

        ds_frame = ds_frame.rename({'value_field': col_name(ds_name)}, axis=1)

        id_name1 = ds_frame.index.name
        id_name2 = merged_frame.index.name

        id_type1 = ds_frame.index.dtype
        id_type2 = merged_frame.index.dtype

        crs1, crs2 = ds_frame.crs, merged_frame.crs

        if id_name1 != id_name2:
            raise AttributeError(
                f"To use this particular merge, all datasets must have the same index name. The function "
                f"received indices with names {id_name1}, and {id_name2}.")

        if id_type1 != id_type2:
            raise AttributeError(
                f"To use this particular merge, all datasets must have the same index type. The function "
                f"received indices with names {id_type1}, and {id_type2}.")

        if crs1 != crs2:
            warnings.warn(f"The datasets passed into this function should have the same crs. "
                          f"The function got {crs1} and {crs2}.")

        if not ds_frame.empty:
            on_list = [id_name1, 'geometry']
            on_list.extend(common_columns)
            keep_list = ['geometry', col_name(ds_name)]
            keep_list.extend(keep_columns)
            if 'value_field' not in ds_frame.columns:
                merged_frame = merged_frame.merge(ds_frame[keep_list], how='outer',
                                                  on=on_list)
            else:
                merged_frame = merged_frame.merge(ds_frame[keep_list], how='outer', on=on_list)
                col_names.append(col_name(ds_name))

            for col in col_names:
                merged_frame[col] = merged_frame[col].fillna(0)
        else:
            if errors == 'raise':
                raise AttributeError("One of the dataframes that was passed is empty.")

    result_name = result_name if result_name is not None else 'merge-op'
    merged_frame[result_name] = 0

    def result_helper(row):
        try:
            result = row[col_names[0]]
            for j in range(1, len(col_names)):
                result = float(merge_op(result, float(row[col_names[j]])))
            return result
        except (IndexError, TypeError, ValueError):
            return 0

    merged_frame[result_name] = merged_frame.apply(result_helper, axis=1)

    if drop:
        merged_frame = merged_frame.drop(columns=col_names)

    if crs is not None:
        merged_frame.set_crs(crs, inplace=True)

    return merged_frame


def remove_other_geometries(gdf: GeoDataFrame, geom_type: str) -> GeoDataFrame:
    """Removes unwanted geometry from the GeoDataFrame.

    :param gdf: The GeoDataFrame to remove from
    :type gdf: GeoDataFrame
    :param geom_type: The geometry type that is to be kept
    :type geom_type: str
    :return: The GeoDataFrame without any other geometries
    :rtype: GeoDataFrame
    """

    gdf['gtype'] = gdf.geom_type
    return gdf[gdf['gtype'] == geom_type].drop(columns='gtype')


def simple_geojson(gdf: GeoDataFrame, value_field: Optional[str] = None, id_field: Optional[str] = None,
                   file_output: Optional[str] = None) -> FeatureCollection:
    """Provides a GeoJSON representation of a GeoDataFrame.

    Only works on GeoDataFrames with id, value, and geometry columns.


    :param gdf: The geodataframe to make a GeoJSON of
    :type gdf: GeoDataFrame
    :param value_field: The value column of the dataframe
    :type value_field: str
    :param id_field: An id field to use for the GeoJSON
    :type id_field: Optional[str]
    :param file_output: Filepath for file output
    :type file_output: Optional[str]
    :return: A GeoJSON-like representation of the geodataframe
    :rtype: FeatureCollection
    """
    list_features = []

    value_getter = (lambda r: r[value_field]) if value_field else (lambda r: 0)

    for i, row in gdf.iterrows():
        feature = Feature(id=i if id_field is None else row[id_field], properties={'value': value_getter(row)},
                          geometry=row.geometry)
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else:
        return feat_collection


def clip_hexes_to_hexes(hexes: GeoDataFrame, clip: GeoDataFrame):
    if not hexes.empty and not clip.empty:
        clip = GeoDataFrame(clip['geometry'], geometry='geometry')
        geodf_contains = gpd.sjoin(hexes, clip, op='intersects', how='inner')  # may also be contains
        return geodf_contains
    return hexes


def clip_hexes_to_polygons(hexes: GeoDataFrame, clip: GeoDataFrame) -> GeoDataFrame:
    if not hexes.empty and not clip.empty:
        clip = GeoDataFrame(clip['geometry'], geometry='geometry', crs=clip.crs)
        geodf = gpd.sjoin(hexes.copy(deep=True), clip, op='intersects', how='inner')
        try:
            geodf.set_index('hex', inplace=True)
        except KeyError:
            pass
        return geodf
    return hexes


def clip_points_to_polygons(points: GeoDataFrame, clip: GeoDataFrame) -> GeoDataFrame:
    if not points.empty and not clip.empty:
        clip = GeoDataFrame(clip['geometry'], geometry='geometry', crs=clip.crs)
        geodf = gpd.sjoin(points, clip, op='within', how='inner')
        try:
            geodf.set_index('hex', inplace=True)
        except KeyError:
            pass
        return geodf
    return points


def convert_crs(left: GeoDataFrame, right: GeoDataFrame, crs: Any = 'EPSG:4326'):
    """Converts two GeoDataFrames to the same crs.

    :param left: The left dataframe
    :type left: GeoDataFrame
    :param right: The right dataframe
    :type right: GeoDataFrame
    :param crs: The crs that the two dataframes will be converted to
    :type crs: Any
    """
    if left.crs is None:
        left = left.set_crs(crs)
    else:
        left = left.to_crs(crs=crs)

    if right.crs is None:
        right = right.set_crs(crs)
    else:
        right = right.to_crs(crs=crs)

    return left, right


def sjoinclip(clip: GeoDataFrame, to: GeoDataFrame,
              operation: str = 'intersects',
              enforce_crs: Any = 'EPSG:4326') -> GeoDataFrame:
    """Clips a GeoDataFrame to another via GeoPandas spatial join operations.

    The operation first converts the two GeoDataFrames to the same crs.

    :param clip: The dataframe that is being clipped to the other dataframes boundary
    :type clip: GeoDataFrame
    :param to: The dataframe that is acting like the boundary for the clipping
    :type to: GeoDataFrame
    :param operation: The operation to be performed in the spatial join
    :type operation: str
    :param enforce_crs: The crs to enforce before clipping
    :type enforce_crs: Any
    :return: The clipped dataframe
    :rtype: GeoDataFrame
    """

    clip, to = convert_crs(clip, to, crs=enforce_crs)
    return gpd.sjoin(clip, to[['geometry']], how='inner', op=operation).drop(columns='index_right', errors='ignore')


def gpdclip(clip: GeoDataFrame, to: GeoDataFrame, enforce_crs: Any = 'EPSG:4326',
            keep_geom_type: bool = True):
    """Clips a GeoDataFrame to another via GeoPandas clip function.

    The operation first converts the two dataframes to the same crs.

    :param clip: The dataframe that is being clipped to the other dataframes boundary
    :type clip: GeoDataFrame
    :param to: The dataframe that is acting like the boundary for the clipping
    :type to: GeoDataFrame
    :param enforce_crs: The crs to enforce before clipping
    :type enforce_crs: Any
    :param keep_geom_type: Passed into geopandas clip function
    :type keep_geom_type: bool
    :return: The clipped dataframe
    :rtype: GeoDataFrame
    """

    clip, to = convert_crs(clip, to, crs=enforce_crs)
    return gpd.clip(clip, to, keep_geom_type=keep_geom_type)


# this function needs to be changed a little bit to work with any dataframe
def generate_grid_over(gdf: GeoDataFrame, hex_resolution: int) -> GeoDataFrame:
    """This function generates a hexagonal grid around a dataframe (a box).

    :param gdf: The dataframe to generate a hex-box around
    :type gdf: GeoDataFrame
    :param hex_resolution: The resolution to use for the grid
    :type hex_resolution: int
    :return: The resulting grid box
    :rtype: GeoDataFrame
    """
    return bin_by_hexid(hexify_dataframe(GeoDataFrame(geometry=[Polygon.from_bounds(*gdf.total_bounds)],
                                                      crs="EPSG:4326"),
                                         hex_resolution=hex_resolution), binning_fn=lambda lst: 0, add_geoms=True)


def find_center_simple(col):
    gs = GeoSeries(col, dtype='float64')
    return gs.unary_union.centroid


def find_ranges_simple(col):
    gs = GeoSeries(col, dtype='float64')
    bounds = gs.total_bounds
    return (bounds[0], bounds[2]), (bounds[1], bounds[3])
