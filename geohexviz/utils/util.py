import collections.abc
import os.path as pth
from typing import Dict, Any, List, Iterable, Union
from collections import defaultdict
import numpy as np
import collections
from copy import deepcopy

from pandas import concat
import pandas as pd
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


def get_percdiff(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def parse_args_kwargs(item, default_args=None, default_kwargs=None):
    if default_args is None:
        default_args = ()
    if default_kwargs is None:
        default_kwargs = {}

    if isinstance(item, dict):
        try:
            return item['args'], item['kwargs']
        except KeyError:
            try:
                return default_args, item['kwargs']
            except KeyError:
                try:
                    return item['args'], default_kwargs
                except KeyError:
                    if item:
                        return default_args, item
            return default_args, default_kwargs
    elif isinstance(item, str):
        return [item], default_kwargs
    elif isinstance(item, list):
        return item, default_kwargs
    else:
        return default_args, default_kwargs


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


def dict_deep_update(d: dict, u: dict) -> object:
    """Updates a dict without changing nested dicts that may be present.

    :param d: The dict to update
    :type d: dict
    :param u: The updating dict
    :type u: dict
    """
    for k, v in u.items():
        if isinstance(d, collections.abc.Mapping):
            if isinstance(v, collections.abc.Mapping):
                r = dict_deep_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d


def get_occurrences(lst: list, **kwargs) -> list:
    """Retrieves a list of tuples containing the list item and its frequency in the list.

    :param lst: The list to count frequencies from
    :type lst: list
    :param kwargs: Keyword arguments for the sorted function
    :type kwargs: **kwargs
    :return: The list of item frequency pairs
    :rtype: list
    """
    occ = list(sorted(((x, lst.count(x)) for x in set(lst)), key=lambda item: item[1], **kwargs))
    return [list(group) for key, group in groupby(occ, itemgetter(1))]


def get_sorted_best(lst: list, allow_ties: bool = True, join_ties: bool = True, selector: Iterable = None,
                    reverse: bool = True) -> object:
    """Retrieves the best entry or entries from a list of labels based on occurrences.

    :param lst: The list of labels to parse the best or worst option from
    :type lst: list
    :param allow_ties: Whether to allow ties between labels or not
    :type allow_ties: bool
    :param join_ties: Whether to join the labels of the tie if present or not
    :type join_ties: bool
    :param selector: If multiple items are tied, this determines the order of which they will be selected as the best
    :type selector: Iterable
    :param reverse: Whether to reverse the list or not (reversed=best, !reversed=worst)
    :type reverse: bool
    :return: The best or worst option from the list
    :rtype: object
    """
    if selector is None:
        selector = []

    occ = get_occurrences(lst, reverse=reverse)

    def select(lst: list):
        if len(selector) > 0:
            for select in selector:
                if select in lst:
                    return select
        return lst[0]

    for i in range(0, len(occ)):
        group = [occ[i][j][0] for j in range(len(occ[i]))]
        while np.nan in group:
            group.remove(np.nan)
        while 'empty' in group:
            group.remove('empty')
        if len(group) == 0:
            group = [np.nan]

        occ[i] = (' & '.join(sorted(group)) if join_ties else 'tie') if allow_ties else select(group) \
            if len(group) > 1 else group[0]
    return occ[0]


def get_best(*args, **kwargs) -> object:
    """Gets the best option from a list of labels.

    see get_sorted_best()

    :param args: Arguments to be passed into the get_sorted_best() function
    :type args: *args
    :param kwargs: Keywords to be passed into the get_sorted_best() function
    :type kwargs: **kwargs
    :return: The best option in the list
    :rtype: object
    """
    return get_sorted_best(*args, reverse=True, **kwargs)


def get_worst(*args, **kwargs) -> object:
    """Gets the worst option from a list of labels.

    see get_sorted_best()

    :param args: Arguments to be passed into the get_sorted_best() function
    :type args: *args
    :param kwargs: Keywords to be passed into the get_sorted_best() function
    :type kwargs: **kwargs
    :return: The worst option in the list
    :rtype: object
    """
    return get_sorted_best(*args, reverse=False, **kwargs)


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
    try:
        eq_area = gdf.to_crs('EPSG:8857', inplace=False)
    except ValueError:
        return {}
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


def simplify_dicts(fields: dict = None, **kwargs):
    if fields is None:
        fields = {}
    fields.update(kwargs)
    return fields


def get_hex_stats(gdf: GeoDataFrame, assume_same_size: bool = True):
    info = {'hex-count': len(gdf), 'hex-total-area': get_total_hex_area(gdf, assume=assume_same_size)}
    return info


def get_column_or_default(df: DataFrame, col: str, default_val=None):
    try:
        return df[col]
    except KeyError:
        return default_val


def get_column_type(df: DataFrame, col: str) -> str:
    if all(isinstance(x, (int, float)) for x in df[col]):
        return 'NUM'
    elif all(isinstance(x, str) or pd.isna(x) for x in df[col]):
        return 'STR'
    return 'UNK'


def generate_random_ids(n: int):
    gids = set()
    while len(gids) != n:
        gids.add(np.random.randint(0, n))
    return gids


def generate_dataframe_random_ids(df: DataFrame):
    df['r-ids'] = generate_random_ids(len(df))
    return df
