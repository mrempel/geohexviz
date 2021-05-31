import numpy as np
from typing import Sequence, Union, Set, Dict, Any, Optional
import math

import geopandas as gpd
from geojson import MultiLineString
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from . import colorscales as cli

world_shape_definitions = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_shape_definitions['name'] = world_shape_definitions['name'].apply(lambda s: s.upper())
world_shape_definitions['continent'] = world_shape_definitions['continent'].apply(lambda s: s.upper())

def get_shapes_from_world(name: Optional[str] = None) -> GeoDataFrame:
    if name:
        if name in world_shape_definitions['continent'].values:
            return world_shape_definitions[world_shape_definitions['continent'] == name].reset_index()
        elif name in world_shape_definitions['name'].values:
            return world_shape_definitions[world_shape_definitions['name'] == name].reset_index()
        else:
            raise KeyError(f"Could not find world object, got={name}.")
    else:
        return world_shape_definitions.copy()


def logify_info(values: Union[Sequence[float], Set[float]], text_type: str = 'raw',
                exp_type: Optional[str] = None,
                fill_first: bool = True, fill_last: bool = True,
                include_min: bool = False, include_max: bool = False,
                minmax_rounding: int = 6,
                min_prefix: str = '', min_suffix: str = '',
                max_prefix: str = '', max_suffix: str = '') -> Dict[str, Any]:
    """Retrieves a dictionary of information for a log scale.

    entries:
    scale-min -> The minimum value on the scale
    scale-max -> The maximum value on the scale
    original-values -> The values given
    logged-values -> The values after having log10 performed on them
    scale-vals -> The scale values
    scale-text -> The text for each scale values


    :param values: The values to compute a log scale for
    :type values: Union[Sequence[float], Set[float]]
    :param text_type: Determines how to format the tick text, (latex, html, raw)
    :type text_type: Optional[str]
    :param exp_type: What type of exponent to select (None, E, ^, *, r)
    :type exp_type: Optional[str]
    :param fill_first: Whether to extend the first segment of the log scale to the next lowest power or not
    :type fill_first: bool
    :param fill_last: Whether to extend the last segment of the log scale to the next highest power or not
    :type fill_last: bool
    :param include_min: Whether to include a value-text pair for the minimum value
    :type include_min: bool
    :param include_max: Whether to include a value-text pair for the maximum value
    :type include_max: bool
    :param minmax_rounding: The number of decimals to round the min and max values to
    :type minmax_rounding: int
    :param min_prefix: Prefix for the minimum value-text pair
    :type min_prefix: str
    :param min_suffix: Suffix for the minimum value-text pair
    :type min_suffix: str
    :param max_prefix: Prefix for the maximum value-text pair
    :type max_prefix: str
    :param max_suffix: Suffix for the maximum value-text pair
    :type max_suffix: str
    :return: A dictionary of the above information
    :rtype: dict
    """
    text_type = str(text_type).lower()
    if str(exp_type).lower() not in ['none', 'e', '^', '*', 'r']:
        raise TypeError("You must select a exponent type within (None, E/e, ^, r).")
    if text_type not in ['latex', 'html', 'raw']:
        raise TypeError("You must select a text type within (latex, html, raw).")

    info = {'original-values': list(values)}
    info['logged-values'] = list(np.log10(info['original-values']))

    info['scale-min'] = round(min(info['logged-values']), minmax_rounding)
    info['scale-max'] = round(max(info['logged-values']), minmax_rounding)
    info['scale-vals'] = list(range(int(info['scale-min']), int(info['scale-max']) + 1))

    min_differs = info['scale-vals'][0] != info['scale-min']
    max_differs = info['scale-vals'][-1] != info['scale-max']

    if min_differs:
        if fill_first:
            info['scale-vals'].insert(0, int(math.floor(info['scale-min'])))
        if include_min:
            info['scale-vals'].insert(0, info['scale-min'])

    if max_differs:
        if fill_last:
            info['scale-vals'].append(int(math.ceil(info['scale-max'])))

        if include_max:
            info['scale-vals'].append(info['scale-max'])

    info['scale-vals'].sort()

    if text_type == 'html':
        if str(exp_type).lower() == 'e':
            info['scale-text'] = [f'<span>1{exp_type}{x}</span>' for x in info['scale-vals']]
        elif exp_type == '^':
            info['scale-text'] = [f'<span>10<sup>{x}</sup></span>' for x in info['scale-vals']]
        elif exp_type == '*':
            info['scale-text'] = [f'<span>10**{x}</span>' for x in info['scale-vals']]
        elif exp_type == 'r':
            info['scale-text'] = [f'<span>10^{x}</span>' for x in info['scale-vals']]
        else:
            info['scale-text'] = [f'<span>{pow(10, x)}</span>' for x in info['scale-vals']]
    elif text_type in 'latex':

        if str(exp_type).lower() == 'e':
            info['scale-text'] = [f'$1{exp_type}{x}$' for x in info['scale-vals']]
        elif exp_type == '^':
            info['scale-text'] = [f'$10^{x}$' for x in info['scale-vals']]
        elif exp_type == '*':
            info['scale-text'] = [f'$10**{x}$' for x in info['scale-vals']]
        else:
            info['scale-text'] = [f'${pow(10, x)}$' for x in info['scale-vals']]

    elif text_type in 'raw':

        if str(exp_type).lower() == 'e':
            info['scale-text'] = [f'1{exp_type}{x}' for x in info['scale-vals']]
        elif exp_type == '^':
            info['scale-text'] = [f'10^{x}' for x in info['scale-vals']]
        elif exp_type == '*':
            info['scale-text'] = [f'10**{x}' for x in info['scale-vals']]
        else:
            info['scale-text'] = [f'{pow(10, x)}' for x in info['scale-vals']]

    try:
        min_loc = info['scale-vals'].index(info['scale-min'])
        if not include_min and min_differs:
            info['scale-text'][min_loc] = ''
        else:
            info['scale-text'][min_loc] = f'{min_prefix}{info["scale-text"][min_loc]}{min_suffix}'
    except ValueError:
        pass
    try:
        max_loc = info['scale-vals'].index(info['scale-max'])
        if not include_max and max_differs:
            info['scale-text'][max_loc] = ''
        else:
            info['scale-text'][max_loc] = f'{max_prefix}{info["scale-text"][max_loc]}{max_suffix}'
    except ValueError:
        pass

    info['scale'] = list(zip(info['scale-vals'], info['scale-text']))

    return info


def logify_scale(df: DataFrame, **kwargs) -> Dict[str, Any]:
    """Converts a manager into log scale form.

    :param df: The dataframe that contains the values for the scale
    :type df: DataFrame
    :param kwargs: Keyword arguments to be passed into logify_info
    :type kwargs: **kwargs
    """

    scale_info = logify_info(df['value_field'], **kwargs)
    df['value_field'] = scale_info['logged-values']

    return {'colorbar': {'tickvals': scale_info['scale-vals'], 'ticktext': scale_info['scale-text']},
            'zmin': scale_info['scale-vals'][0], 'zmax': scale_info['scale-vals'][-1]}


def conformOpacity(properties: Dict[str, Any], conform_alpha: bool = True):
    """Conforms the opacity of a colorscale to match the opacity of the plotly marker.

    :param properties: The dict of plotly properties
    :type properties: Dict[str, Any]
    :param conform_alpha: Whether to conform the opacity or not
    :type conform_alpha: bool
    """
    if conform_alpha:
        try:
            alpha = properties['marker']['opacity']
        except KeyError:
            alpha = 1.0

        properties['colorscale'] = cli.configureScaleWithAlpha(properties['colorscale'], alpha=alpha)


def geopolygons_to_points(gdf: GeoDataFrame):
    nodes = gpd.GeoDataFrame(columns=list(gdf.columns))
    # Extraction of the polygon nodes and attributes values from polys and integration into the new GeoDataFrame
    i = 0
    gdf['POLY_NUM'] = 0
    for index, row in gdf.iterrows():
        for j in list(row['geometry'].exterior.coords):
            nodes = nodes.append(
                {'POLY_NUM': i, 'geometry': Point(j)},
                ignore_index=True)
        i += 1
    return nodes


def linestring_to_points(ls, columns):
    nodes = gpd.GeoDataFrame(columns=columns)
    for x, y in ls.coords.xy:
        nodes.append({'geometry': Point(y, x)}, ignore_index=True)
    return nodes


def polygon_to_points(poly, columns):
    nodes = gpd.GeoDataFrame(columns=columns)
    for j in list(poly.exterior.coords):
        nodes = nodes.append(
            {'geometry': Point(j)},
            ignore_index=True)
    return nodes


def geos_to_points(gdf: GeoDataFrame, set_index: bool = True, errors: str = 'raise'):
    nodes = gpd.GeoDataFrame(columns=list(gdf.columns))
    if not gdf.empty:
        i = 0
        gdf['POLY_NUM'] = 0
        for index, row in gdf.iterrows():
            shape = row.geometry
            if isinstance(shape, LineString):
                points = linestring_to_points(shape, list(gdf.columns))
            elif isinstance(shape, MultiLineString):
                points = gpd.GeoDataFrame(columns=list(gdf.columns))
                for ls in shape:
                    points = points.append(
                        linestring_to_points(ls, list(gdf.columns)), ignore_index=True)
            elif isinstance(shape, Polygon):
                points = polygon_to_points(shape, list(gdf.columns))
            elif isinstance(shape, MultiPolygon):
                points = gpd.GeoDataFrame(columns=list(gdf.columns))
                for mp in shape:
                    points = points.append(
                        polygon_to_points(mp, list(gdf.columns)), ignore_index=True)
            elif isinstance(shape, Point):
                points = gpd.GeoDataFrame(columns=list(gdf.columns))
                points = points.append({'geometry': shape}, ignore_index=True)
            else:
                raise AttributeError(f"The shape can not be converted into points, shape={shape}.")

            for col in gdf.columns:
                if col not in ['POLY_NUM', 'geometry']:
                    points[col] = row[col]
            points['POLY_NUM'] = i
            nodes = nodes.append(points, ignore_index=True)
            i += 1
        if set_index:
            nodes.set_index('POLY_NUM', drop=True, inplace=True)
        return nodes
    if errors == 'raise':
        raise ValueError("The given dataframe must not be empty.")

    return nodes


def to_plotly_points_format(gdf: GeoDataFrame, disjoint: bool = True):
    gdf['gtype'] = gdf.geom_type.astype(str)
    notpoints = gdf[gdf['gtype'] != 'Point']
    notpoints = geos_to_points(notpoints, errors='ignore')

    notpoints = notpoints.append(gdf[gdf['gtype'] == 'Point'], ignore_index=False)

    notpoints.index.set_names('POLY_NUM', inplace=True)
    notpoints.drop(columns='gtype', inplace=True)

    lats = []
    lons = []

    for i in notpoints.index.unique():
        poly = notpoints[notpoints.index == i]
        lats.extend(list(poly.geometry.y))
        lons.extend(list(poly.geometry.x))
        if disjoint:
            lats.append(np.nan)
            lons.append(np.nan)

    return lats, lons
