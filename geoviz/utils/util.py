import collections.abc
import os.path as pth
from typing import Dict, Any, List, Iterable, Union
from collections import defaultdict
import numpy as np
import collections
from copy import deepcopy

from pandas import concat
from geopandas import GeoDataFrame
from itertools import groupby
from operator import itemgetter
from h3 import h3
from pandas import DataFrame
from pandas.api.types import is_hashable, is_numeric_dtype, is_string_dtype
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint, MultiLineString, GeometryCollection

DataSet = Dict[str, Any]
GeometryContainer = Union[MultiPolygon, MultiPoint, MultiLineString, GeometryCollection]
AnyGeom = Union[Polygon, Point, LineString, GeometryContainer]

def fix_filepath(filepath: str, add_filename: str = '', add_ext: str = '') -> str:
    """Converts a directorypath, or filepath into a valid filepath.

    :param filepath: The filepath to convert
    :type filepath: str
    :param add_filename: The filename to add if there is none
    :type add_filename: str
    :param add_ext: The extension to add if there is a filename
    :type add_ext: str
    :return: The converted filepath
    :rtype: str
    """
    add_ext = add_ext.replace('.', '')
    if pth.isdir(filepath):
        return pth.join(filepath, f"{add_filename}.{add_ext}" if add_ext else add_filename)
    else:
        _, extension = pth.splitext(filepath)
        if not extension:
            return f"{filepath}.{add_ext}" if add_ext else filepath
        return filepath


def dict_deep_update(d: dict, u: dict):
    """Updates a dict without changing nested dicts that may be present.

    :param d: The dict to update
    :type d: dict
    :param u: The updating dict
    :type u: dict
    """
    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = dict_deep_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d


def get_occurrences(lst: list, **kwargs):
    occ = list(sorted(((x, lst.count(x)) for x in set(lst)), key=lambda item: item[1], **kwargs))
    return [list(group) for key, group in groupby(occ, itemgetter(1))]


def get_sorted_occurrences(lst: list, allow_ties=False, join_ties=True, selector=[], **kwargs):
    occ = get_occurrences(lst, **kwargs)

    def select(lst: list):
        if len(selector) > 0:
            for select in selector:
                if select in lst:
                    return select
        return lst[0]

    for i in range(len(occ)):
        for j in range(len(occ[i])):
            occ[i][j] = str(occ[i][j][0])
        group = occ[i]
        if isinstance(group, list):
            group = list(sorted(group))
            while np.nan in group:
                group.remove(np.nan)
            while 'empty' in group:
                group.remove('empty')
            if not allow_ties:
                group = select(group)
            else:
                if join_ties:
                    group = ','.join(group)
                else:
                    group = 'tie'
        occ[i] = group
    return occ[0]


def sorted_indices(data: Iterable, **kwargs) -> List:
    """Sorts indices of a list, ignoring zeroes.

    :param data: The data to sort by index
    :type data: Iterable
    :param kwargs: Keyword arguments to pass into the sorted function
    :type kwargs: **kwargs
    :return: A list of sorted indices
    :rtype: List
    """
    indices = defaultdict(list)
    for i, x in enumerate(data):
        indices[x].append(i)
    unique_sorted_data = sorted({x for x in data if x != 0}, **kwargs)
    return [y[0] if len(y) == 1 else y for y in (indices[x] for x in unique_sorted_data)]


def make_multi_dataset(dss, errors: str = 'raise'):
    vals = []
    for dsname, ds in dss.items():
        ds['data']['*DS_NAME*'] = dsname
        vals.append(ds['data'])
    try:
        return {'data': GeoDataFrame(concat(vals, ignore_index=True), geometry='geometry', crs='EPSG:4326')}
    except ValueError as e:
        if errors == 'raise':
            raise e
        data = GeoDataFrame()
        data['*DS_NAME*'] = ''
        return {'data': data}


def dissolve_multi_dataset(mds, properties, dropmds: bool = True, mdsname: str = '*COMBINED*', errors: str = 'raise'):
    mds = mds['data']
    dss = {}
    for un in mds['*DS_NAME*'].unique():
        try:
            try:
                dss[un] = deepcopy(properties[un])
            except KeyError:
                dss[un] = {}
            dss[un].update({'data': mds[mds['*DS_NAME*'] == un]})
        except KeyError as e:
            if errors == 'raise':
                raise e
            pass

    if dropmds:
        try:
            properties.pop(mdsname)
        except KeyError as e:
            if errors == 'raise':
                raise e
            pass
    return dss


def get_total_hex_area(gdf: GeoDataFrame, assume: bool = True):
    total = 0

    area = lambda poly: h3.hex_area(h3.h3_get_resolution(poly))

    if not gdf.empty:
        if assume:
            total = area(gdf.index.values[0]) * len(gdf)
        else:
            for i, row in gdf.iterrows():
                total += area(i)
    return total


def get_stats(gdf: GeoDataFrame, hexed: bool = False):
    eq_area = gdf.to_crs('EPSG:8857', inplace=False)
    if hexed:
        estimated_area = get_total_hex_area(gdf)
    else:
        estimated_area = sum(eq_area.geometry.apply(lambda p: p.area / 10 ** 6))
    estimated_bounds = eq_area.unary_union.bounds
    info = {
        'geometries-count': len(gdf),
        'estimated-area': estimated_area,
        'estimated-bounds': estimated_bounds
    }
    return info


def get_hex_stats(gdf: GeoDataFrame, assume_same_size: bool = True):
    info = {'hex-count': len(gdf), 'hex-total-area': get_total_hex_area(gdf, assume=assume_same_size)}
    return info


def geom_all_type(gdf: GeoDataFrame, wanted_type: str):
    types = set(list(gdf.geom_type))
    if len(types) == 1:
        return True if str(types.pop()) == wanted_type else False
    return False


def rename_dataset(dataset: DataSet):
    df = dataset.pop('data')
    for item in dataset.items():
        if is_hashable(item[1]) and item[1] in df.columns:
            df.rename({item[1]: item[0]}, axis=1, inplace=True)

    dataset['data'] = df


def get_column_or_default(df: DataFrame, col: str, default_val=None):
    try:
        return df[col]
    except KeyError:
        return default_val


def get_column_type(df: DataFrame, col: str):
    col = get_column_or_default(df, col)
    if col is not None:
        if is_numeric_dtype(col):
            return 'num'
        elif is_string_dtype(col):
            return 'str'
    return 'unk'


def generate_random_ids(n: int):
    gids = set()
    while len(gids) != n:
        gids.add(np.random.randint(0, n))
    return gids


def generate_dataframe_random_ids(df: DataFrame):
    df['r-ids'] = generate_random_ids(len(df))
    return df
