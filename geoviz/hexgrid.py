# imports
import json
import math
import operator
import warnings
from typing import Callable, List, Optional, Dict, Any, Union, Tuple, Iterable
from collections import defaultdict
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from geojson import Feature, FeatureCollection, MultiPolygon
from geopandas import GeoDataFrame, GeoSeries
from h3 import h3
from pandas import DataFrame
from pandas.api.types import is_hashable, is_numeric_dtype, is_string_dtype
from shapely.geometry import Polygon, Point, polygon, LineString, MultiLineString, GeometryCollection, \
    MultiPoint

DataSet = Dict[str, Any]
DataSets = Dict[str, DataSet]

'''
Notes:
Polygons must have Long/Lat form in order to be plotted with GeoJSONs.
'''

GeometryContainer = Union[MultiPolygon, MultiPoint, MultiLineString, GeometryCollection]
AnyGeom = Union[Polygon, Point, LineString, GeometryContainer]


"""
COMBINING WITH ORIGINAL FUNCTIONALITY TO MAKE IT BETTER.
____________________________________________________________________
"""


def validate_dataframe_hexes(df: DataFrame, hex_field: Optional[str] = None, store_validity: bool = True) -> bool:
    if hex_field:
        hex_field = df[hex_field]
    else:
        hex_field = pd.Index.to_series(index=df.index, name='pind')

    if store_validity:
        isvalid, df['hex-validity'] = validate_hexes(hex_field, get_sequence=True)
    else:
        isvalid = validate_hexes(hex_field, get_sequence=False)
    return isvalid


def validate_hexes(hexids: Iterable, get_sequence: bool = True):
    validity = [h3.h3_is_valid(x) for x in hexids]

    if get_sequence:
        return (False not in validity), validity
    else:
        return False not in validity


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
        if _check_antimeridian(p0[0], p1[0]):
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


def check_crossing(lon1: float, lon2: float, validate: bool = True):
    """
    Assuming a minimum travel distance between two provided longitude coordinates,
    checks if the 180th meridian (antimeridian) is crossed.
    """
    if validate and any(abs(x) > 180.0 for x in [lon1, lon2]):
        raise ValueError("longitudes must be in degrees [-180.0, 180.0]")
    return abs(lon2 - lon1) > 180.0


def hexify_geodataframe(gdf: GeoDataFrame, hex_resolution: int = 3) -> GeoDataFrame:
    """Makes a new GeoDataFrame, with the index set as the hex cell ids that each geometry in the geometry column
    corresponds to.

    :param gdf: The GeoDataFrame to place a hex cell id overlay on
    :type gdf: GeoDataFrame
    :param hex_resolution: The resolution of the hexes to be generated
    :type hex_resolution: int
    :return: A GeoDataFrame with a hex id index
    :rtype: GeoDataFrame
    """
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
        elif isinstance(shape, LineString):
            return [shape_to_hex_ids(Point(x))[0] for x in shape.coords]
        else:
            return get_multi_shape_hex(shape, shape_to_hex_ids)

    cdf['hex'] = cdf.geometry.apply(shape_to_hex_ids)
    cdf['hexlen'] = cdf['hex'].apply(len)
    return pd.DataFrame.explode(cdf[cdf['hexlen'] != 0].drop(columns='hexlen').reset_index(drop=True),
                                'hex', ignore_index=True).set_index('hex', drop=True)


def hexify_geometry(gdf: GeoDataFrame, hex_col: Optional[str] = None) -> GeoDataFrame:
    """Adds the geometry of the hex ids in the given column or index.

    :param gdf: The GeoDataFrame containing the hex ids
    :type gdf: GeoDataFrame
    :param hex_col: The column containing hex ids
    :type hex_col: Optional[str]
    :return: A GeoDataFrame with hex ids and their geometries
    :rtype: GeoDataFrame
    """
    if hex_col:
        gdf.geometry = gdf[hex_col].apply(lambda r: Polygon(h3.h3_to_geo_boundary(r, True)))
    else:
        gdf['hexv'] = gdf.index.values
        gdf.geometry = gdf['hexv'].apply(lambda r: Polygon(h3.h3_to_geo_boundary(r, True)))
        gdf.drop(columns='hexv', inplace=True)

    return gdf


def bin_by_hex(hex_gdf: GeoDataFrame, binning_fn: Callable, *binning_args, hex_field: Optional[str] = None,
               binning_field: Optional[str] = None,
               binned_name: str = 'items', result_name: str = 'value_field', add_geoms: bool = False,
               **binning_kw) -> GeoDataFrame:
    """Bins a DataFrame by hex cell ids.

    This function assumes the dataframe has a VALID hex and geometry columns.
    Using these columns the data is grouped into what hexagon on the grid they fall into.


    :param hex_gdf: A dataframe representing any kind of hex data
    :type hex_gdf: GeoDataFrame
    :param binning_field: The column that the data will be grouped by
    :type binning_field: str
    :param binning_fn: The function to perform after grouping
    :type binning_fn: Callable
    :param binned_name: The name of the column that contains the grouped result
    :type binned_name: str
    :param result_name: The name of the column that contains the grouped result after having the function applied
    :type result_name: str
    :param add_geoms: Whether to add hex geometries after grouping or not
    :type add_geoms: bool
    :return: A frame containing the binned hex grid and its geometries
    :rtype: GeoDataFrame
    """

    if not binning_field:
        hex_gdf['binby'] = list(range(0, len(hex_gdf)))
        binning_field = 'binby'

    if hex_field:
        hex_field = hex_gdf[hex_field]
    else:
        hex_field = hex_gdf.index

    # group by ids aggregate into list
    hex_gdf_g = (hex_gdf
                 .groupby(hex_field)[binning_field]
                 .agg(list)
                 .to_frame(binned_name))

    hex_gdf_g[result_name] = (hex_gdf_g[binned_name].apply(binning_fn, args=binning_args, **binning_kw))

    if add_geoms:
        hex_gdf_g = hexify_geometry(GeoDataFrame(hex_gdf_g))
        hex_gdf_g.crs = "EPSG:4326"

    return hex_gdf_g


def _fill_polygon(poly: Polygon, hex_resolution: int = 3):
    poly_fill = h3.polyfill(poly.__geo_interface__, hex_resolution, geo_json_conformant=True)
    return list(poly_fill), [Polygon(h3.h3_to_geo_boundary(y, geo_json=True)) for y in poly_fill]


def get_hex_geodataframe_loss(gdf: GeoDataFrame, hex_resolution: int = 3) -> GeoDataFrame:
    """Generates the hexagons that correspond to the shapes in the dataframe (Point, Polygon only)

    This function can be quite lengthy.

    This function assumes there is a geometry column present in the dataframe. Using
    this column a hexagonal grid is generated with its ids being stored in the hex
    column and its geometry stored in the geometry column.

    :param gdf: The dataframe to convert to a hex dataframe
    :type gdf: GeoDataFrame
    :param hex_resolution: the resolution of the hexagons to be generated (0-16)
    :type hex_resolution: int
    :return: A dataframe containing hexagon ids and geometries
    :rtype: GeoDataFrame
    """

    cpy = gdf.reset_index().explode()
    cpy['hex'] = ''

    new_df_dict = defaultdict(list)

    for i, row in cpy.iterrows():
        shape = row.geometry
        if isinstance(shape, Point):
            get_hex = h3.geo_to_h3(lat=shape.y, lng=shape.x, resolution=hex_resolution)
            try:
                poly = Polygon(h3.h3_to_geo_boundary(get_hex, geo_json=True))
                new_df_dict['geometry'].append(poly)
                new_df_dict['hex'].append(get_hex)
                mask = row.index.isin(['hex', 'geometry'])
                masked = row.loc[~mask]

                for index, value in masked.items():
                    new_df_dict[index].append(value)
            except:
                continue
        elif isinstance(shape, Polygon):
            hexes, geoms = _fill_polygon(shape, hex_resolution=hex_resolution)
            new_df_dict['hex'].extend(hexes)
            new_df_dict['geometry'].extend(geoms)
            mask = row.index.isin(['hex', 'geometry'])
            masked = row.loc[~mask]

            for index, value in masked.items():
                new_df_dict[index].extend([value] * len(hexes))

    new_df = GeoDataFrame(DataFrame(new_df_dict), geometry='geometry', crs="EPSG:4326")
    new_df.set_index('hex', inplace=True)
    new_df['value'] = 0

    return new_df


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


def convert_dataframe_coordinates_to_geodataframe(df: DataFrame, latitude_field: str = 'latitude',
                                                  longitude_field: str = 'longitude') -> GeoDataFrame:
    """Converts a pandas dataframe to a GeoDataFrame based on pre-existing lat/lon fields.

    This function takes a pandas dataframe and converts it by returning a GeoDataFrame
    with its geometry attribute pointing at a new geometry column of the given dataframe.

    :param df: The dataframe to convert
    :type df: DataFrame
    :param latitude_field: The name of the column containing lat values
    :type latitude_field: str
    :param longitude_field: The name of the column containing lon values
    :type longitude_field: str
    :return: The converted dataframe (now GeoDataFrame)
    :rtype: GeoDataFrame
    """
    return GeoDataFrame(df, geometry=gpd.points_from_xy(df[longitude_field], df[latitude_field], crs='EPSG:4326'))


def conform_geodataframe_geometry(gdf: GeoDataFrame, d3_geo: bool = True, fix_polys: bool = True):
    """Conforms the given dataframe's geometry to the given standard.

    This function can be used to obtain a GeoDataFrame that of which has conformed dataframe
    geometry to the given standard (False-> clockwise, True->counterclockwise)

    :param gdf: Any dataframe with a geometry column
    :type gdf: GeoDataFrame
    :param d3_geo: The conforming standard
    :type d3_geo: bool
    :return: The same dataframe with conformed geometry
    :rtype: GeoDataFrame
    """

    gdf.geometry = gdf.geometry.apply(func=conform_polygon, d3_geo=d3_geo, fix_poly=fix_polys)
    return gdf


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
                return MultiPolygon([fn(s) for s in shape])
            except (TypeError, ValueError):
                return shape
        except TypeError:
            return shape

    if fix_polys:
        perform = (lambda s: polygon.orient(_fix_polygon_bounds(s), sign=-1)) if d3_geo else (
            lambda s: polygon.orient(_fix_polygon_bounds(s), sign=1))
    else:
        perform = (lambda s: polygon.orient(s, sign=-1)) if d3_geo else (lambda s: polygon.orient(s, sign=1))

    gdf['valid'] = gdf.is_valid
    geoms = gdf.geometry.apply(conform_geoshape, args=[perform])

    gdf.geometry = list(geoms.values)
    return gdf


def conform_polygon(poly: Polygon, d3_geo: bool = True, fix_poly: bool = True) -> Polygon:
    """Conforms the given polygon to the given standard.

    :param poly: The polygon to conform
    :type poly: Polygon
    :param d3_geo: The conform standard (True->Clockwise,False->Counterclockwise)
    :type d3_geo: bool
    :return: The conformed Polygon
    :rtype: Polygon
    """
    try:
        if fix_poly:
            poly = _fix_polygon_bounds(poly)
        return polygon.orient(poly, sign=-1) if d3_geo else polygon.orient(poly, sign=1)
    except AttributeError:
        if fix_poly:
            perform = (lambda p: polygon.orient(_fix_polygon_bounds(p), sign=-1)) if d3_geo else (
                lambda p: polygon.orient(_fix_polygon_bounds(p), sign=1))
        else:
            perform = (lambda p: polygon.orient(p, sign=-1)) if d3_geo else (lambda p: polygon.orient(p, sign=1))
        lst = []
        for i in range(len(poly)):
            lst.append(perform(poly[i]))
        return MultiPolygon(lst)


def get_area(poly: Polygon):
    geod = Geod(ellps="WGS84")
    return abs(geod.geometry_area_perimeter(poly)[0])


def merge_datasets_simple(datasets: List[Union[GeoDataFrame, Tuple[str, GeoDataFrame]]],
                          merge_op: Callable = operator.add,
                          common_columns: Optional[List[str]] = None, keep_columns: Optional[List[str]] = None,
                          drop: bool = True, crs: Optional[str] = None) -> GeoDataFrame:
    """Merges the datasets with the given merge operation.

    :param datasets: The datasets to merge
    :type datasets: List[Union[GeoDataFrame, Tuple[str, GeoDataFrame]]]
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
    :return: A merged dataframe
    :rtype: GeoDataFrame
    """
    merged_frame = GeoDataFrame(geometry=[])
    first = datasets[0]
    if isinstance(first, tuple):
        if not isinstance(first[1], GeoDataFrame):
            raise AttributeError(f"If you input a tuple, the first item must be the datasets name and the second a "
                                 f"GeoDataFrame object. The function got an incorrect tuple: {first}")
        merged_frame.index.set_names(first[1].index.name, inplace=True)
        merged_frame.set_crs(crs=first[1].crs, inplace=True)
    else:
        merged_frame.index.set_names(first.index.name, inplace=True)
        merged_frame.set_crs(crs=first.crs, inplace=True)

    col_name = lambda i: 'vf-' + str(i)
    col_names = []
    ds_num = 1

    common_columns = common_columns if common_columns else []
    keep_columns = keep_columns if keep_columns else []

    for item in datasets:

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
            raise AttributeError(f"To use this particular merge, all datasets must have the same name. The function "
                                 f"received indices with names {id_name1}, and {id_name2}.")

        if id_type1 != id_type2:
            raise AttributeError(f"To use this particular merge, all datasets must have the same name. The function "
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
            raise AttributeError("One of the dataframes that was passed is empty.")

    merged_frame['merge-op'] = 0
    for i, row in merged_frame.iterrows():
        try:
            result = row[col_names[0]]
            for j in range(1, len(col_names)):
                result = float(merge_op(result, float(row[col_names[j]])))

            merged_frame.at[i, 'merge-op'] = result
        except IndexError:
            break
        except TypeError:
            break
        except ValueError:
            break

    if drop:
        merged_frame = merged_frame.drop(columns=col_names)

    if crs is not None:
        merged_frame.set_crs(crs, inplace=True)

    return merged_frame


def _get_occurrences_dict(lst: List) -> Dict[str, int]:
    """Counts the occurrences of each item in the list.

    :param lst: The list to find the occurrences in
    :type lst: List
    :return: A dictionary of each term and their frequency
    :rtype: Dict[str, int]
    """

    return {x: lst.count(x) for x in lst}


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
    gdf = gdf[gdf['gtype'] == geom_type]
    gdf.drop(columns='gtype', inplace=True)
    return gdf


def geodataframe_to_geojson(gdf: GeoDataFrame, value_field: str, id_field: Optional[str] = None,
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

    for i, row in gdf.iterrows():
        feature = Feature(id=i if id_field is None else row[id_field], properties={'value': row[value_field]},
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
        geodf = gpd.sjoin(hexes, clip, op='intersects', how='inner')
        try:
            geodf.set_index('hex', inplace=True)
        except KeyError:
            pass
        print('GEODF', geodf)
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


def generate_grid_over_hexes(gdf: GeoDataFrame, hex_column: Optional[str] = None):
    if hex_column is None:
        gdf['hex_resolution_col'] = gdf.index
        hex_resolution_col = gdf['hex_resolution_col'].astype(str).apply(h3.h3_get_resolution)
    else:
        hex_resolution_col = gdf[hex_column].apply(h3.h3_get_resolution)
    hex_resolutiones = _get_occurrences_dict(list(hex_resolution_col))
    hes_res = int(max(hex_resolutiones.items(), key=operator.itemgetter(1))[0])

    range_lon, range_lat = find_ranges(gdf.geometry)
    range_lat, range_lon = list(range_lat), list(range_lon)

    bl = [range_lon[0], range_lat[0]]
    br = [range_lon[1], range_lat[0]]
    tl = [range_lon[0], range_lat[1]]
    tr = [range_lon[1], range_lat[1]]

    # TODO: Polygon must be in lat/lng format

    poly = Polygon([bl, tl, tr, br, bl])

    gdf = GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    hexed_gdf = hexify_geodataframe(gdf, hex_resolution=hes_res)
    return conform_geogeometry(hexed_gdf, fix_polys=True)


def find_center_simple(col):
    gs = GeoSeries(col, dtype='float64')
    return gs.unary_union.centroid


def find_center(col: List[Union[Point, Polygon]]):
    """Finds the center of a given list of geometry.

    :param col: The geometry to find the center of
    :type col: List[Union[Point,Polygon]]
    :return: The center point
    :rtype: Point
    """
    x_list = []
    y_list = []
    for s in col:
        if isinstance(s, Point):
            x_list.append(s.x)
            y_list.append(s.y)
        elif isinstance(s, Polygon):
            for p in s.exterior.coords:
                x_list.append(p[1])
                y_list.append(p[0])

    # center_x = sum(x_list) / len(x_list)  # (max(x_list) - min(x_list)) / 2
    # center_y = sum(y_list) / len(y_list)  # (max(y_list) - min(y_list)) /2
    try:
        center_x = (max(x_list) + min(x_list)) / 2
        center_y = (max(y_list) + min(y_list)) / 2
        return Point(center_y, center_x)
    except ValueError:
        return None


def find_ranges_simple(col):
    gs = GeoSeries(col, dtype='float64')
    bounds = gs.total_bounds
    return (bounds[0], bounds[2]), (bounds[1], bounds[3])


def find_ranges(col):
    x_list = []
    y_list = []

    for s in col:
        if isinstance(s, Point):
            x_list.append(s.x)
            y_list.append(s.y)
        elif isinstance(s, Polygon):
            for p in s.exterior.coords:
                x_list.append(p[0])
                y_list.append(p[1])

    try:
        range_x = min(x_list), max(x_list)
        range_y = min(y_list), max(y_list)
        return range_x, range_y
    except ValueError:
        return None


'''
_______________________________________________________________________________
This section provides some functions that may be applied to dataframes for
analysis purposes.
_______________________________________________________________________________
'''

"""
Prints a DataFrame or GeoDataFrame along with some important information
"""


def print_dataframe_info(df: GeoDataFrame, name='Dataframe'):
    print(name, '|LEN: ' + str(len(df)) + '|TYPE: ' + str(type(df)) + '|\n', df.head())


def apply_dist(row, hex_id_field, lat, lon):
    point = h3.h3_to_geo(row[hex_id_field])
    return h3.point_dist((lat, lon), (point.y, point.x))


def apply_reachable(row, hex_id_field, lat, lon, time, speed):
    point = h3.h3_to_geo(row[hex_id_field])
    dist = h3.point_dist((lat, lon), (point.y, point.x))
    return 1 if dist / speed <= time else 0


def apply_time_to(row, hex_id_field, lat, lon, speed, max_time):
    # point = h3.h3_to_geo(row[hex_id_field])
    point = row.geometry.centroid
    dist = calculate_latlon_distance(lat, lon, point.y, point.x)
    # dist = h3.point_dist((lat, lon), (point.y, point.x))

    return dist / speed if (dist / speed) <= max_time else 0


"""
Uses the Heaverside function to calculate the distance
between two lat/lon coordinates.
"""


def calculate_latlon_distance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def lat_lng_2_theta_phi(lat, lng):
    theta = 90 - lat
    if lng > 0:
        phi = lng
    else:
        phi = 360 - lng
    return theta, phi


def polar_2_cart(theta, phi, radius=1):
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    return x, y, z


def deg_2_rad(deg):
    return math.pi * deg / 180.


# determine the winding of a polygon where
# pc is the center of the polygon
# p0 is the first point in the list
# p1 is the second point in the list
# all points originally in lat/lng
def is_winding_ccw(pc, p0, p1):
    # convert the points from lat/lng to polar
    pcPolar = lat_lng_2_theta_phi(pc[1], pc[0])
    p0Polar = lat_lng_2_theta_phi(p0[1], p0[0])
    p1Polar = lat_lng_2_theta_phi(p1[1], p1[0])
    # convert from polar to cartesian
    pcCart = polar_2_cart(pcPolar[0], pcPolar[1])
    p0Cart = polar_2_cart(p0Polar[0], p0Polar[1])
    p1Cart = polar_2_cart(p1Polar[0], p1Polar[1])
    # determine the winding direction
    pd1 = (p0Cart[0] - pcCart[0], p0Cart[1] - pcCart[1], p0Cart[2] - pcCart[2])
    pd2 = (p1Cart[0] - pcCart[0], p1Cart[1] - pcCart[1], p1Cart[2] - pcCart[2])
    crss = np.cross(pd1, pd2)
    w = np.dot(crss, pcCart)
    if w > 0:
        return True
    else:
        return False


def determine_orient(poly: Polygon) -> bool:
    coords = list(poly.exterior.coords)
    centroid = list(poly.centroid.coords)
    return is_winding_ccw(centroid[0], coords[0], coords[1])


def get_best(lst):
    maxitem = max(lst)
    best_lst = []
    for i in range(0, len(lst)):
        if lst[i] == maxitem:
            best_lst.append(i)
    return best_lst
