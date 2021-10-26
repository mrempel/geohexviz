import numpy as np
from typing import Sequence, Union, Set, Dict, Any, Optional, List, Tuple
import math

import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import DataFrame
from . import colorscales as cli
from .colorscales import get_scale, configureScaleWithAlpha, configure_color_opacity
import geohexviz.utils.geoutils as gcg
import pandas as pd

world_shape_definitions = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_shape_definitions['name'] = world_shape_definitions['name'].apply(lambda s: s.upper())
world_shape_definitions['continent'] = world_shape_definitions['continent'].apply(lambda s: s.upper())


def get_shapes_from_world(name: Optional[str] = None) -> GeoDataFrame:
    """Retrieves a continent or country from the world, or the whole world.

    This function uses GeoPandas naturalearth_lowres dataset.
    If the given name is None, the function will retrieve the shapes of thw entire world.

    :param name: The name of the continent or country to get the shapes for
    :type name: str
    :return: The dataframe containing the geometry of the country, continent, or world
    :rtype: GeoDataFrame
    """
    if name:
        if name in world_shape_definitions['continent'].values:
            return world_shape_definitions[world_shape_definitions['continent'] == name].reset_index()
        elif name in world_shape_definitions['name'].values:
            return world_shape_definitions[world_shape_definitions['name'] == name].reset_index()
        else:
            raise KeyError(f"Could not find world object, got={name}.")
    else:
        return world_shape_definitions.copy(deep=True)


def format_latex_exp10(exp: float, exp_type: str) -> str:
    """Formats a exponent to the power of 10 in latex form.
    """
    if exp_type.lower() not in ['e', '^', '*', 'r']:
        raise ValueError(f"The 'exp_type' argument must be one of ['e', '^', '*', 'r']. Received {exp_type}.")

    if str(exp_type).lower() == 'e':
        return f'$1{exp_type}{exp}$'
    elif exp_type == '^':
        return f'$10^{exp}$'
    elif exp_type == '*':
        return f'$10**{exp}$'
    else:
        return f'${pow(10, exp)}$'


def format_html_exp10(exp: float, exp_type: str) -> str:
    """Formats a exponent to the power of 10 in html form.
    """
    if exp_type.lower() not in ['e', '^', '*', 'r', 'n']:
        raise ValueError(f"The 'exp_type' argument must be one of ['e', '^', '*', 'r', 'n']. Received {exp_type}.")

    if str(exp_type).lower() == 'e':
        return f'<span>1{exp_type}{exp}</span>'
    elif exp_type == '^':
        return f'<span>10<sup>{exp}</sup></span>'
    elif exp_type == '*':
        return f'<span>10**{exp}</span>'
    elif exp_type == 'r':
        return f'<span>10^{exp}</span>'
    else:
        return f'<span>{pow(10, exp)}</span>'


def format_raw_exp10(exp: float, exp_type: str) -> str:
    """Formats a exponent to the power of 10 in raw form.
    """
    if exp_type.lower() not in ['e', '^', '*', 'r']:
        raise ValueError(f"The 'exp_type' argument must be one of ['e', '^', '*', 'r']. Received {exp_type}.")

    if str(exp_type).lower() == 'e':
        return f'1{exp_type}{exp}'
    elif exp_type == '^':
        return f'10^{exp}'
    elif exp_type == '*':
        return f'10**{exp}'
    else:
        return f'{pow(10, exp)}'


expmap10 = {
    'latex': format_latex_exp10,
    'html': format_html_exp10,
    'raw': format_raw_exp10
}


def logify_info(values: Union[Sequence[float], Set[float]], text_type: str = 'raw',
                exp_type: Optional[str] = None,
                fill_last: bool = True,
                include_min: bool = False, include_max: bool = False,
                minmax_rounding: int = 3, include_predecessors: bool = False,
                min_prefix: str = '', min_suffix: str = '',
                max_prefix: str = '', max_suffix: str = '') -> Dict[str, Any]:
    """Retrieves a dictionary of information for a log scale.

    entries:
    scale-min -> The minimum value on the scale
    scale-max -> The maximum value on the scale
    original-values -> The values given
    logged-values -> The values after having log10 performed on them
    scale-dict -> The scale values and their text

    :param values: The values to compute a log scale for
    :type values: Union[Sequence[float], Set[float]]
    :param text_type: Determines how to format the tick text, (latex, html, raw)
    :type text_type: Optional[str]
    :param exp_type: What type of exponent to select (None, E, ^, *, r)
    :type exp_type: Optional[str]
    :param fill_last: Whether to extend the last segment of the log scale to the next highest power or not
    :type fill_last: bool
    :param include_min: Whether to include a value-text pair for the minimum value
    :type include_min: bool
    :param include_max: Whether to include a value-text pair for the maximum value
    :type include_max: bool
    :param minmax_rounding: The number of decimals to round the min and max values to
    :type minmax_rounding: int
    :param include_predecessors: Whether to include all previous exponents in the scale values or not
    :type include_predecessors: bool
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

    if exp_type is None:
        exp_type = '^'
    try:
        expfn = expmap10[str(text_type).lower()]
    except KeyError:
        raise ValueError(f"The 'text_type' argument must be one of {list(expmap10.keys())}.")

    logged = np.array(list(np.log10(list(values))))
    l = logged[logged >= 0]
    info = {'original-values': list(values), 'logged-values': list(logged), 'scale-min': round(min(l), minmax_rounding),
            'scale-max': round(max(l), minmax_rounding), 'scale-dict': {}}

    if fill_last:
        last = int(math.ceil(info['scale-max']))
        info['scale-dict'][last] = expfn(last, exp_type)
    start = int(info['scale-min'])
    end = int(info['scale-max']) + 1
    scale = list(range(0, start)) if include_predecessors else []
    scale.extend(list(range(start, end)))

    if len(scale) == 1 and 0 not in scale:
        scale.insert(0, 0)

    for i in scale:
        info['scale-dict'][i] = expfn(i, exp_type)

    if include_min:
        info['scale-dict'][info['scale-min']] = f"{min_prefix}{expfn(info['scale-min'], exp_type)}{min_suffix}"

    if include_max:
        info['scale-dict'][info['scale-max']] = f"{max_prefix}{expfn(info['scale-max'], exp_type)}{max_suffix}"

    info['scale-dict'] = dict(sorted(info['scale-dict'].items(), key=lambda it: it[0]))
    return info


def logify_info_dep(values: Union[Sequence[float], Set[float]], text_type: str = 'raw',
                    exp_type: Optional[str] = None,
                    fill_first: bool = True, fill_last: bool = True,
                    include_min: bool = False, include_max: bool = False,
                    minmax_rounding: int = 6,
                    min_prefix: str = '', min_suffix: str = '',
                    max_prefix: str = '', max_suffix: str = '') -> Dict[str, Any]:
    """Retrieves a dictionary of information for a log scale.

    :DEPRECATED:

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
            newfirst = int(math.floor(info['scale-min']))
            if newfirst not in info['scale-vals']:
                info['scale-vals'].insert(0, int(math.floor(info['scale-min'])))
        if include_min:
            info['scale-vals'].insert(0, info['scale-min'])

    if max_differs:
        if fill_last:
            newlast = int(math.ceil(info['scale-max']))
            if newlast not in info['scale-vals']:
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

    tv, tt = list(zip(*list(scale_info['scale-dict'].items())))
    result = {'colorbar': {'tickvals': tv, 'ticktext': tt},
              'zmin': tv[0], 'zmax': tv[-1]}
    return result


def logify_scale_dep(df: DataFrame, **kwargs) -> Dict[str, Any]:
    """Converts a manager into log scale form.

    :DEPRECATED:

    :param df: The dataframe that contains the values for the scale
    :type df: DataFrame
    :param kwargs: Keyword arguments to be passed into logify_info
    :type kwargs: **kwargs
    """

    scale_info = logify_info(df['value_field'], **kwargs)
    df['value_field'] = scale_info['logged-values']

    return {'colorbar': {'tickvals': scale_info['scale-vals'], 'ticktext': scale_info['scale-text']},
            'zmin': scale_info['scale-vals'][0], 'zmax': scale_info['scale-vals'][-1]}


def opacify_colorscale(dataset: dict, alpha: float = None):
    """Plot wrapper for adjusting the opacity of a colorscale.

    :param dataset: The dataset whose colorscale to adjust
    :type dataset: dict
    :param alpha: The alpha to adjust by (if not, then the alpha within the dataset is used)
    :type alpha: float
    """
    colorscale = dataset['manager'].get('colorscale')

    try:
        colorscale = get_scale(colorscale)
    except AttributeError:
        pass
    opacity = dataset['manager'].get('marker', {}).pop('opacity', 1)
    alpha = alpha if alpha is not None else opacity
    dataset['manager']['colorscale'] = cli.configure_cscale_opacity(colorscale, alpha)


# TODO: monitor bugs with the clipping functions (maybe merge "to" before)

def sjoin_clip(clip: GeoDataFrame, to: List[GeoDataFrame], operation: str = 'intersects', validate: bool = False) -> GeoDataFrame:
    """Plot wrapper for the spatial join clip operation.

    :param clip: The data to clip
    :type clip: GeoDataFrame
    :param to: The list of data to act as the clip
    :type to: List[GeoDataFrame]
    :param operation: The operation to be performed in the spatial join
    :type operation: str
    :param validate: Whether or not to validate the result or not (throws)
    :type validate: bool
    :return: The result of the clip
    :rtype: GeoDataFrame
    """
    if len(to) == 0:
        raise ValueError("There are no datasets to clip to.")

    result = pd.concat(
        [gcg.sjoinclip(clip, item, operation=operation) for item in to], axis=0).drop_duplicates()

    if validate:
        cgtypes = gcg.get_present_geomtypes(clip, allow_collections=True, collapse_geoms=True)
        if not all(gt in cgtypes for gt in gcg.get_present_geomtypes(
                result, allow_collections=True, collapse_geoms=True)):
            raise ValueError("The clipped dataframe has geometry types that were not present in the original.")

    return result


def gpd_clip(clip: GeoDataFrame, to: List[GeoDataFrame], validate: bool = True):
    """Plot wrapper for the geopandas clip operation.

    :param clip: The data to clip
    :type clip: GeoDataFrame
    :param to: The list of data to act as the clip
    :type to: List[GeoDataFrame]
    :param validate: Whether or not to validate the result or not (throws)
    :type validate: bool
    :return: The result of the clip
    :rtype: GeoDataFrame
    """
    if len(to) == 0:
        raise ValueError("There are no datasets to clip to.")
    result = pd.concat(
        [gcg.gpdclip(clip, item) for item in to]).drop_duplicates()

    if validate:
        cpointlike = gcg.check_geom_only(clip, 'Point', collapse_geoms=True)
        if not cpointlike:
            if any('Point' in gcg.get_present_geomtypes(gt) for gt in to):
                raise TypeError("Non-point-like geometries can not be clipped to point-like geometries.")

        cgtypes = gcg.get_present_geomtypes(clip, allow_collections=True, collapse_geoms=True)
        if not all(gt in cgtypes for gt in gcg.get_present_geomtypes(
                result, allow_collections=True, collapse_geoms=True)):
            raise ValueError("The clipped dataframe has geometry types that were not present in the original.")

        # There may be an error if the user somehow gets the clip result to not be the same type as the original clip
    return result
