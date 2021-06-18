from copy import deepcopy
from typing import Any, Tuple, ClassVar, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from os import path
from os.path import join as pjoin

import plotly.colors
from plotly.graph_objs import Figure, Choropleth, Scattergeo, Choroplethmapbox, Scattermapbox
from shapely.geometry import Point, Polygon

from geoviz.utils.util import fix_filepath, get_sorted_occurrences, generate_dataframe_random_ids, get_column_type, \
    simplify_dicts, dict_deep_update
from geoviz.utils import geoutils as gcg
from geoviz.utils import plot_util as butil

from geoviz.utils.colorscales import solid_scale, configureColorWithAlpha, configureScaleWithAlpha, getDiscreteScale, \
    tryGetScale

from geopandas import GeoDataFrame
from pandas import DataFrame
from collections import defaultdict

_extension_mapping = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    '.shp': gpd.read_file,
    '': gpd.read_file
}

_group_functions = {
    'sum': lambda lst: sum(lst),
    'avg': lambda lst: sum(lst) / len(lst),
    'min': lambda lst: min(lst),
    'max': lambda lst: max(lst),
    'count': lambda lst: len(lst),
    'bestworst': get_sorted_occurrences
}


def _prepare_choropleth_trace(gdf: GeoDataFrame, mapbox: bool = False):
    geojson = gcg.simple_geojson(gdf, 'value_field')

    if mapbox:
        return Choroplethmapbox(
            locations=gdf.index,
            z=gdf['value_field'],
            geojson=geojson
        )
    else:
        return Choropleth(
            locations=gdf.index,
            z=gdf['value_field'],
            geojson=geojson
        )


def _prepare_scattergeo_trace(gdf: GeoDataFrame, separate: bool = True, disjoint: bool = False, mapbox: bool = False):
    lats = []
    lons = []

    if separate:
        for polynum in gdf.index.unique():
            df = gdf[gdf.index == polynum]
            lats.extend(list(df.geometry.y))
            lons.extend(list(df.geometry.x))
            if disjoint:
                lats.append(np.nan)
                lons.append(np.nan)
        if disjoint:
            try:
                lats.pop()
                lons.pop()
            except IndexError:
                pass
    else:
        if disjoint:
            for _, row in gdf.iterrows():
                lats.append(row.geometry.y)
                lons.append(row.geometry.x)
                lats.append(np.nan)
                lons.append(np.nan)

            try:
                lats.pop()
                lons.pop()
            except IndexError:
                pass
        else:
            lats = list(gdf.geometry.y)
            lons = list(gdf.geometry.x)

    if mapbox:
        return Scattermapbox(
            lat=lats,
            lon=lons
        )
    else:
        return Scattergeo(
            lat=lats,
            lon=lons
        )


"""
Notes:

Clipping has significant margin of error.
For this application we should not allow Polygon or MultiPolygon datasets to be clipped to Points.
Check CRS before clipping.
We can either use geopandas.clip() or geopandas.sjoin(), however we have limited experience with clip() and 
sjoin() fails a lot.
clip() tends to fail when clipping to a weird geometry type such as points.

We should also optimize how parameters are passed into the reading functions.

Our custom functions main functionality should be put into a separate module and imported, maybe?
"""


def _validate_dataset(name: str, dataset: dict):
    if '+' in name:
        raise ValueError("There must not be a '+' in the name of a dataset.")

    if 'data' not in dataset:
        raise ValueError("There must be a 'data' member present in the dataset.")


def _read_dataset(name: str, dataset: dict, default_manager: dict = None, allow_manager_updates: bool = True):
    _validate_dataset(name, dataset)
    data = dataset['data']

    if default_manager is None:
        default_manager = {}

    if isinstance(data, str):
        filepath, extension = path.splitext(pjoin(path.dirname(__file__), data))
        filepath = fix_filepath(filepath, add_ext=extension)

        try:
            data = _extension_mapping[extension](filepath)
        except KeyError:
            pass
            # logger.warning("The general file formats accepted by this application are (.csv, .shp). Be careful.")

    if isinstance(data, GeoDataFrame) or isinstance(data, DataFrame):

        if data.empty:
            raise ValueError("If the data passed is a DataFrame, it must not be empty.")

        data = data.copy(deep=True)

        if 'geometry' not in data.columns:

            try:
                latitude_field = data[dataset.pop('latitude_field')]
            except KeyError:
                if 'latitude_field' in data.columns:
                    latitude_field = data['latitude_field']
                else:
                    raise ValueError(
                        "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                        "and longitude_field entries. Missing latitude_field member.")

            try:
                longitude_field = data[dataset.pop('longitude_field')]
            except KeyError:
                if 'longitude_field' in data.columns:
                    longitude_field = data['longitude_field']
                else:
                    raise ValueError(
                        "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                        "and longitude_field entries. Missing longitude_field member.")

            dataset['data'] = data = GeoDataFrame(data, geometry=gpd.points_from_xy(longitude_field, latitude_field,
                                                                                    crs='EPSG:4326'))
    else:
        raise ValueError("The 'data' member of the dataset must only be a string, DataFrame, or GeoDataFrame object.")

    input_manager = dataset.pop('manager', {})
    dataset['manager'] = {}
    _update_manager(dataset, **default_manager)
    if allow_manager_updates:
        _update_manager(dataset, **input_manager)
    elif input_manager:
        raise ValueError("This dataset may not have a custom manager.")


def _update_manager(dataset: dict, updates: dict = None, override: bool = False, **kwargs):
    updates = simplify_dicts(fields=updates, **kwargs)
    if override:
        dataset['manager'] = updates
    else:
        dict_deep_update(dataset['manager'], updates)


def _hexify_data(data, hex_resolution: int):
    return gcg.hexify_geodataframe(data, hex_resolution=hex_resolution)


def _bin_by_hex(data, *args, binning_field: str = None, binning_fn=None, **kwargs):
    if binning_fn in _group_functions:
        binning_fn = _group_functions[binning_fn]

    if binning_field is None:
        vtype = 'num'
    else:
        vtype = get_column_type(data, binning_field)

    if binning_fn is None:
        binning_fn = _group_functions['bestworst'] if vtype == 'str' else _group_functions['count']

    return gcg.conform_geogeometry(
        gcg.bin_by_hex(data, binning_fn, *args, binning_field=binning_field, result_name='value_field',
                       add_geoms=True, **kwargs)), vtype


def _hexify_dataset(dataset: dict, hex_resolution: int):
    dataset['data'] = _hexify_data(dataset['data'], dataset.pop('hex_resolution', hex_resolution))


def _bin_dataset_by_hex(dataset: dict):
    binning_fn = dataset.pop('binning_fn', None)
    binning_args = dataset.pop('binning_args', ())
    binning_kw = dataset.pop('binning_kwargs', {})

    if isinstance(binning_fn, dict):
        if ('args' in binning_fn and binning_args) or ('kwargs' in binning_fn and binning_kw):
            raise ValueError("Only one set of binning arguments and keywords may be passed.")

        binning_args = binning_fn.get('args', binning_args)
        binning_kw = binning_fn.get('kwargs', binning_kw)
        try:
            binning_fn = binning_fn['fn']
        except KeyError:
            raise ValueError(
                "If binning function is passed as a dict, there must be a valid 'fn' entry denoting function.")

    dataset['data'], dataset['VTYPE'] = _bin_by_hex(dataset['data'], *binning_args,
                                                    binning_field=dataset.pop('binning_field', None),
                                                    binning_fn=binning_fn,
                                                    **binning_kw)


def _hexbinify_data(data, hex_resolution: int, *args, binning_field: str = None, binning_fn=None, **kwargs):
    data = _hexify_data(data, hex_resolution)
    return _bin_by_hex(data, *args, binning_field=binning_field, binning_fn=binning_fn, **kwargs)


def _hexbinify_dataset(dataset: dict, hex_resolution: int):
    _hexify_dataset(dataset, hex_resolution)
    _bin_dataset_by_hex(dataset)


def _create_dataset(data, fields: dict = None, **kwargs) -> dict:
    if fields is None:
        fields = {}
    fields.update(dict(data=data, **kwargs))
    return fields


def _split_name(name: str) -> Tuple[str, str]:
    lind = name.index(':')
    return name[:lind], name[lind + 1:]


def _update_helper(dataset: dict, updates: dict, override: bool = False):
    if override:
        dataset['manager'] = updates
    else:
        dataset['manager'].update(updates)  # change to dict deep update


def _prepare_general_datasetv2(data, fields: dict = None, allow_manager_updates: bool = False, **kwargs):
    print()


# this function should standalone.
def _prepare_general_dataset(name: str, dataset: dict, **kwargs):
    try:
        dataset['data'] = butil.get_shapes_from_world(dataset['data'])
    except (KeyError, ValueError, TypeError):
        # logger.debug("If name was a country or continent, the process failed.")
        pass

    _read_dataset(name, dataset, **kwargs)
    dataset['data'] = gcg.conform_geogeometry(dataset['data'], fix_polys=True)[['geometry']]
    # logger.debug('dataframe geometry conformed to GeoJSON standard.')

    if dataset.pop('to_boundary', False):
        dataset['data'] = gcg.unify_geodataframe(dataset['data'])
    dataset['data']['value_field'] = 0
    dataset['VTYPE'] = 'num'


class PlotBuilder:
    """This class contains a Builder implementation for visualizing Plotly Hex data.
        """

    # default choropleth manager (includes properties for choropleth only)
    _default_dataset_manager: ClassVar[Dict[str, Any]] = dict(
        colorscale='Viridis',
        colorbar=dict(
            title='COUNT'
        ),
        marker=dict(
            line=dict(
                color='white',
                width=0.60
            ),
            opacity=0.8
        ),
        hoverinfo='location+text'
    )

    # default scatter plot manager (includes properties for scatter plots only)
    _default_point_manager: ClassVar[Dict[str, Any]] = dict(
        mode='markers+text',
        marker=dict(
            line=dict(
                color='black',
                width=0.3
            ),
            color='white',
            symbol='circle-dot',
            size=20
        ),
        showlegend=False,
        textposition='top center',
        textfont=dict(
            color='Black',
            size=5
        )
    )

    _default_grid_manager: ClassVar[Dict[str, Any]] = dict(
        colorscale=[[0, 'white'], [1, 'white']],
        zmax=1, zmin=0,
        marker=dict(
            line=dict(
                color='black'
            ),
            opacity=0.2
        ),
        showlegend=False,
        showscale=False,
        hoverinfo='text'
    )

    _default_region_manager: ClassVar[Dict[str, Any]] = dict(
        colorscale=[[0, 'rgba(255,255,255,0.525)'], [1, 'rgba(255,255,255,0.525)']],
        marker=dict(
            line=dict(
                color="rgba(0,0,0,1)",
                width=0.65
            )
        ),
        legendgroup='regions',
        zmin=0,
        zmax=1,
        showlegend=False,
        showscale=False,
        hoverinfo='text'
    )

    _default_outline_manager: ClassVar[Dict[str, Any]] = dict(
        mode='lines',
        line=dict(
            color="black",
            width=1,
            dash='dash'
        ),
        legendgroup='outlines',
        showlegend=False,
        hoverinfo='text'
    )

    # contains the default properties for the figure
    _default_figure_manager: ClassVar[Dict[str, Any]] = dict(
        geos=dict(
            projection=dict(
                type='orthographic'
            ),
            showcoastlines=False,
            showland=True,
            landcolor="rgba(166,166,166,0.625)",
            showocean=True,
            oceancolor="rgb(222,222,222)",
            showlakes=False,
            showrivers=False,
            showcountries=False,
            lataxis=dict(
                showgrid=True
            ),
            lonaxis=dict(
                showgrid=True
            )
        ),
        layout=dict(
            title=dict(
                text='',
                x=0.5
            ),
            margin=dict(
                r=0, l=0, t=0, b=0
            )
        )
    )

    _default_regions: ClassVar[Dict[str, Dict]] = dict(canada=dict(
        data='CANADA'
    ))

    _default_grids: ClassVar[Dict[str, Dict]] = dict(canada=dict(
        data='CANADA'
    ))

    _default_plot_settings: ClassVar[Dict[str, Any]] = {
        'hex_resolution': 3,
        'plot_regions': True,
        'plot_grids': True,
        'plot_points': True,
        'plot_outlines': True,
        'range_buffer_lat': (0.0, 0.0),
        'range_buffer_lon': (0.0, 0.0),
        'plot_output_service': 'plotly'
    }

    def __init__(self, main=None, regions=None, grids=None, outlines=None, points=None):

        self._figure = Figure()
        self.update_figure_manager(**self._default_figure_manager)

        self._container = {
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        # grids will all reference this manager
        self._grid_manager = deepcopy(self._default_grid_manager)

        self._plot_settings = deepcopy(self._default_plot_settings)

        if main is not None:
            self.set_main(**main)

        if regions is not None:
            for k, v in regions.items():
                self.add_region(k, **v)

        if grids is not None:
            for k, v in grids.items():
                self.add_grid(k, **v)

        if outlines is not None:
            for k, v in outlines.items():
                self.add_outline(k, **v)

        if points is not None:
            for k, v in points.items():
                self.add_point(k, **v)

    # may not be necessary
    def __setitem__(self, key, value):
        if key == 'main':
            self.set_main(**value)
        else:
            try:
                typer, name = _split_name(key)
            except ValueError:
                raise ValueError("The given string should be in the form of '<type>:<name>'.")

            if typer == 'region':
                self.add_region(name, **value)
            elif typer == 'grid':
                self.add_grid(name, **value)
            elif typer == 'outline':
                self.add_outline(name, **value)
            elif typer == 'point':
                self.add_point(name, **value)
            else:
                raise ValueError(f"The given dataset type does not exist. Must be one of ['region', 'grid', "
                                 f"'outline', 'point']. Received {typer}.")

    # may not be necessary (needs to be fixed)
    def __delitem__(self, key):
        if key in ['regions', 'grids', 'outlines', 'points', 'main']:
            self._container[key] = {}
        else:
            try:
                typer, name = _split_name(key)

            except ValueError:
                raise ValueError("The given string should be one of ['regions', 'grids', 'outlines', 'points', "
                                 "'main'] or in the form of '<type>:<name>'.")
            try:
                cont = self._container[f'{typer}s']
            except KeyError:
                raise ValueError(f"The given dataset type does not exist. Must be one of ['region', 'grid', "
                                 f"'outline', 'point']. Received {typer}.")
            try:
                del cont[name]
            except KeyError:
                raise ValueError(f"The dataset with the name ({name}) could not be found within {typer}s.")

    # we don't have to use getattr
    def __getitem__(self, item):
        return self.search(item)

    """
    MAIN DATASET FUNCTIONS
    """

    def set_main(self, data, fields: dict = None, **kwargs):
        _read_dataset('MAIN', dataset := _create_dataset(data, fields=fields, **kwargs),
                      default_manager=deepcopy(self._default_dataset_manager))
        _hexbinify_dataset(dataset, 3)
        dataset['DSTYPE'] = 'MN'
        self._container['main'] = dataset

    def get_main(self):
        try:
            return self._container['main']
        except KeyError:
            raise KeyError(f"The main dataset could not be found.")

    def remove_main(self):
        try:
            del self._container['main']
        except KeyError:
            raise KeyError("The main dataset could not be found.")

    def update_main_manager(self, updates: dict = None, override: bool = False, **kwargs):
        updates = simplify_dicts(fields=updates, **kwargs)
        _update_helper(self._container['main'], updates, override)

    """
    REGION FUNCTIONS
    """

    def add_region(self, name: str, data, fields: dict = None, **kwargs):
        """Adds a region-type dataset to the builder.

        Region-type datasets should consist of Polygon-like geometries.
        Best results are read from a GeoDataFrame, or DataFrame.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param fields: Additional information for this dataset
        :type fields: Dict[str, Any]
        :param kwargs: Additional fields for this dataset
        :type kwargs: **kwargs
        """
        _prepare_general_dataset(name, dataset := _create_dataset(data, fields=fields, **kwargs),
                                 default_manager=deepcopy(self._default_region_manager),
                                 allow_manager_updates=True)
        if 'Point' in dataset['data'].geom_type.values:
            raise NotImplementedError("Can not have region geometry type as points.")
        dataset['DSTYPE'] = 'RGN'
        self.get_regions()[name] = dataset

    def get_region(self, name: str):
        try:
            return self.get_regions()[name]
        except KeyError:
            raise KeyError(f"The region-type dataset with the name '{name}' could not be found.")

    def get_regions(self):
        return self._container['regions']

    def remove_region(self, name: str):
        try:
            del self.get_regions()[name]
        except KeyError:
            raise ValueError(f"The region-type dataset with the name '{name}' could not be found.")

    def update_region_manager(self, name: str = None, updates: dict = None, override: bool = False, **kwargs):
        updates = simplify_dicts(fields=updates, **kwargs)

        if name is None:
            for _, v in self.get_regions().items():
                _update_helper(v, updates, override)
        else:
            _update_helper(self.get_region(name), updates, override)

    def reset_regions(self):
        self._container['regions'] = {}

    """
    GRID FUNCTIONS
    """

    def add_grid(self, name: str, data, **kwargs):
        _prepare_general_dataset(name, dataset := _create_dataset(data, **kwargs), default_manager=self._grid_manager,
                                 allow_manager_updates=False)
        _hexbinify_dataset(dataset, 3)  # change to default hex res
        dataset['DSTYPE'] = 'GRD'
        self.get_grids()[name] = dataset

    def get_grid(self, name: str):
        try:
            return self.get_grids()[name]
        except KeyError:
            raise KeyError(f"The grid-type dataset with the name '{name}' could not be found.")

    def get_grids(self):
        return self._container['grids']

    def remove_grid(self, name: str):
        try:
            grids = self.get_grids()
            del grids[name]
            if len(grids) == 0:
                self.reset_grids()
        except KeyError:
            raise ValueError(f"The grid-type dataset with the name '{name}' could not be found.")

    def update_grid_manager(self, updates: dict = None, override: bool = False, **kwargs):
        updates = simplify_dicts(fields=updates, **kwargs)
        _update_helper(self._grid_manager, updates, override)

    def reset_grids(self):
        self._container['grids'] = {}
        self._grid_manager = deepcopy(self._default_grid_manager)  # reset to default

    """
    OUTLINE FUNCTIONS
    """

    def add_outline(self, name: str, data, **kwargs):
        _prepare_general_dataset(name, dataset := _create_dataset(data, **kwargs),
                                 default_manager=deepcopy(self._default_outline_manager),
                                 allow_manager_updates=True)
        # consider the pros and cons to converting the dataset to points here
        dataset['DSTYPE'] = 'OUT'
        self.get_outlines()[name] = dataset

    def get_outline(self, name: str):
        try:
            return self.get_outlines()[name]
        except KeyError:
            raise KeyError(f"The outline-type dataset with the name '{name}' could not be found.")

    def get_outlines(self):
        return self._container['outlines']

    def remove_outline(self, name: str):
        try:
            del self.get_outlines()[name]
        except KeyError:
            raise ValueError(f"The outline-type dataset with the name '{name}' could not be found.")

    def update_outline_manager(self, name: str = None, updates: dict = None, override: bool = False, **kwargs):
        if updates is None:
            updates = {}
        updates.update(kwargs)

        if name is None:
            for _, v in self.get_outlines().items():
                _update_helper(v, updates, override)
        else:
            _update_helper(self.get_outline(name), updates, override)

    def reset_outlines(self):
        self._container['outlines'] = {}

    """
    POINT FUNCTIONS
    """

    def add_point(self, name: str, data, **kwargs):
        _prepare_general_dataset(name, dataset := _create_dataset(data, **kwargs),
                                 default_manager=deepcopy(self._default_point_manager),
                                 allow_manager_updates=True)
        dataset['data'] = butil.pointify_geodataframe(dataset['data'])
        dataset['DSTYPE'] = 'PNT'
        self.get_points()[name] = dataset

    def get_point(self, name: str):
        try:
            return self.get_points()[name]
        except KeyError:
            raise KeyError(f"The point-type dataset with the name '{name}' could not be found.")

    def get_points(self):
        return self._container['points']

    def remove_point(self, name: str):
        try:
            del self.get_points()[name]
        except KeyError:
            raise ValueError(f"The point-type dataset with the name '{name}' could not be found.")

    def update_point_manager(self, name: str = None, updates: dict = None, override: bool = False, **kwargs):
        if updates is None:
            updates = {}
        updates.update(kwargs)

        if name is None:
            for _, v in self.get_points().items():
                _update_helper(v, updates, override)
        else:
            _update_helper(self.get_point(name), updates, override)

    def reset_points(self):
        self._container['points'] = {}

    """
    FIGURE FUNCTIONS
    """

    def update_figure_manager(self, updates: dict = None, overwrite: bool = False, **kwargs):
        updates = simplify_dicts(fields=updates, **kwargs)
        self._figure.update_geos(updates.pop('geos', {}), overwrite=overwrite)
        self._figure.update(overwrite=overwrite, **updates)

    """
    DATA ALTERING FUNCTIONS
    """

    def apply_to_query(self, name: str, fn, big_query: bool = True):

        datasets = self.search(name, big_query=big_query)

        lst = []

        if 'data' in datasets:
            lst.append(fn(datasets))
        else:
            for _, v in datasets.items():
                if 'data' in v:
                    lst.append(fn(v))
                else:
                    for _, vv in v.items():
                        if 'data' in vv:
                            lst.append(fn(vv))
                        else:
                            for _, vvv in vv.items():
                                if 'data' in vvv:
                                    lst.append(fn(vvv))
                                else:
                                    raise ValueError("Error when applying function.")
        return lst

    def remove_empties(self, name: str = 'main', empty_symbol: Any = 0, add_to_plot: bool = False):

        if add_to_plot and name != 'main':
            raise ValueError('Empties can only be added to plot if it is the main dataset.')

        def helper(dataset):
            if add_to_plot:
                dataset['empties'] = dataset['empties'].append(
                    dataset['data'][dataset['data']['value_field'] == empty_symbol])
            dataset['data'] = dataset['data'][dataset['data']['value_field'] != empty_symbol]

        self.apply_to_query(name, helper)

    # this is both a data altering, and plot altering function
    def logify_scale(self, name: str = 'main', **kwargs):

        helper = lambda dataset: _update_manager(dataset, butil.logify_scale(dataset['data'], **kwargs))
        self.apply_to_query(name, helper, big_query=True)

    def clip_datasets(self, clip: str, to: str, method: str = 'sjoin', operation: str = None):

        datas = self.apply_to_query(to, lambda ds: ds, big_query=True)

        def helpersjoin(dataset):

            result = pd.concat(
                [gcg.clip(dataset['data'], item['data'], operation=operation, errors='raise') for item in datas])
            result.drop_duplicates(inplace=True)
            dataset['data'] = result

        def helpergpd(dataset):

            alterations = []

            for i, item in enumerate(datas):
                if item['DSTYPE'] == 'PNT' and dataset['DSTYPE'] != 'PNT':
                    raise NotImplementedError(
                        "Cannot clip a non-point dataset to a point dataset. May be implemented in the future.")
                if 'Point' in item['data'].geom_type.values:
                    raise NotImplementedError("Cannot clip to a dataset with point-like geometry.")

                alterations.append(gcg.gpdclip(dataset['data'], item['data']))

            result = pd.concat(alterations)
            result.drop_duplicates(inplace=True)
            dataset['data'] = result

        if method == 'gpd':
            self.apply_to_query(clip, helpergpd, big_query=True)
        elif method == 'sjoin':
            self.apply_to_query(clip, helpersjoin, big_query=True)
        else:
            raise ValueError("The selected method must be one of ['gpd', 'sjoin'].")

    # check methods of clipping
    def simple_clip(self, to: str = 'regions+outlines', method: str = 'sjoin'):
        self.clip_datasets('main+grids', to, method=method, operation='intersects')
        self.clip_datasets('points', to, method=method, operation='within')

    def _remove_underlying_grid(self, df: GeoDataFrame, gdf: GeoDataFrame):
        """Removes the grid from underneath the data (lower file size).

        :param ds: The main dataset (after altreration)
        :type ds: DataSet
        :param gds: The merged grid dataset (after alteration)
        :type gds: DataSet
        """

        if not df.empty and not gdf.empty:
            gdf = gpd.overlay(gdf, df[['value_field', 'geometry']],
                              how='difference')
            gdf = gcg.remove_other_geometries(gdf, 'Polygon')
            return gdf

    """
    PLOT ALTERING FUNCTIONS
    """

    def opacify_colorscale(self, name: str = 'main'):

        def helper(dataset):
            colorscale = dataset['manager'].get('colorscale')

            try:
                colorscale = tryGetScale(colorscale)
            except AttributeError:
                pass

            alpha = dataset['manager'].get('marker', {}).get('opacity', 1)

            if isinstance(colorscale, dict):
                colorscale = {k: configureColorWithAlpha(v) for k, v in colorscale.items()}

            if isinstance(colorscale, tuple):
                colorscale = list(colorscale)
            if isinstance(colorscale, list):
                for i in range(len(colorscale)):
                    if isinstance(colorscale[i], tuple):
                        colorscale[i] = list(colorscale[i])
                    if isinstance(colorscale[i], list):
                        colorscale[i][1] = configureColorWithAlpha(colorscale[i][1], alpha)
                    else:
                        colorscale[i] = configureColorWithAlpha(colorscale[i], alpha)

            dataset['manager']['colorscale'] = configureScaleWithAlpha(colorscale, alpha)

        self.apply_to_query(name, helper)

    def auto_focus(self, on: str = 'main', center_on: bool = False, rotation_on: bool = True, ranges_on: bool = True):

        if not any((center_on, rotation_on, ranges_on)):
            raise ValueError("One of 'center_on', 'roatation_on', 'ranges_on' must be set to True.")

        geoms = []

        def helper(dataset):
            geoms.extend(list(dataset['data'].geometry))

        self.apply_to_query(on, helper)

        if geoms:
            cen: Point = gcg.find_center_simple(geoms)
            rng = gcg.find_ranges_simple(geoms)

            buffer_lat = self._plot_settings['range_buffer_lat']
            buffer_lon = self._plot_settings['range_buffer_lon']

            cd_lat, cd_lon, projection_rotation, center = None, None, None, None

            if rng is not None:
                lat_r = list(rng[1])
                cd_lat = lat_r[0] - buffer_lat[0], lat_r[1] + buffer_lat[1]

                lon_r = list(rng[0])
                cd_lon = lon_r[0] - buffer_lon[0], lon_r[1] + buffer_lon[1]

            if cen is not None:
                projection_rotation = dict(lon=cen.x, lat=cen.y)
                center = dict(lon=cen.x, lat=cen.y)

            geos = {}
            if center_on:
                geos['center'] = center
            if rotation_on:
                geos['projection_rotation'] = projection_rotation
            if ranges_on:
                geos['lataxis_range'] = cd_lat
                geos['lonaxis_range'] = cd_lon

            self._figure.update_geos(**geos)
        else:
            raise ValueError("There are no geometries in your query to focus on.")

    def auto_grid(self, name: str = 'main', by_bounds: bool = False, hex_resolution: int = 3):
        fn = gcg.generate_grid_over_hexes if by_bounds else gcg.hexify_geodataframe

        def helper(dataset):
            return fn(dataset['data'], hex_resolution=hex_resolution)

        try:
            grid = GeoDataFrame(pd.concat(list(self.apply_to_query(name, helper))), crs='EPSG:4326')

            if not grid.empty:
                grid['value_field'] = 0
                self.get_grids()['|*AUTO*|'] = {'data': gcg.conform_geogeometry(grid), 'manager': self._grid_manager}
                print(self.get_grids())
                print(fn)

                import matplotlib.pyplot as plt
                self.get_grids()['|*AUTO*|']['data'].plot()
                plt.show()

        except ValueError as e:
            raise e

    # TODO: handle edge cases for discrete scales.
    def discretize_scale(self, name: str = 'main', scale_type: str = 'sequential', **kwargs):

        def helper(dataset):
            if dataset['VTYPE'] == 'str':
                raise ValueError(f"You can not discretize a qualitative dataset.")
            else:
                low = dataset['manager'].get('zmin', min(dataset['data']['value_field']))
                high = dataset['manager'].get('zmax', max(dataset['data']['value_field']))
                print(dataset['manager'].get('colorscale'))
                dataset['manager']['colorscale'] = getDiscreteScale(dataset['manager'].get('colorscale'), scale_type,
                                                                    low, high, **kwargs)

        self.apply_to_query(name, helper)

    """
    RETRIEVAL/SEARCHING FUNCTIONS
    
    get_regions(), etc... could also fall under here.
    """

    def single_search(self, query: str, big_query: bool = True):

        if query == 'main':
            return self.get_main()
        elif query in ['regions', 'grids', 'outlines', 'points', 'all']:
            if big_query:
                return self._container if query == 'all' else self._container[query]
            else:
                raise ValueError("The given query should not refer to a collection of datasets.")

        try:
            typer, name = _split_name(query)
        except ValueError:
            raise ValueError("The given query should be one of ['regions', 'grids', 'outlines', 'points', "
                             f"'main'] or in the form of '<type>:<name>'. Received item: {query}.")

        if typer == 'region':
            return self.get_region(name)
        elif typer == 'grid':
            return self.get_grid(name)
        elif typer == 'outline':
            return self.get_outline(name)
        elif typer == 'point':
            return self.get_point(name)
        else:
            raise ValueError("The given dataset type does not exist. Must be one of ['region', 'grid', "
                             "'outline', 'point']. "
                             f"Received {typer}.")

    def search(self, query: str, big_query: bool = True):

        sargs = query.split('+')
        if len(sargs) == 1:
            return self.single_search(sargs[0], big_query=big_query)
        else:
            return {k: self.single_search(k, big_query=big_query) for k in sargs}

    get_query_data = lambda self, name: self.apply_to_query(name, lambda ds: ds['data'])

    """
    PLOTTING FUNCTIONS
    """

    def plot_regions(self):
        """Plots the region datasets within the builder.

        All of the regions are treated as separate plot traces.
        """
        # logger.debug('adding regions to plot.')
        for regname, regds in self.get_regions().items():
            choro = _prepare_choropleth_trace(regds['data'],
                                              mapbox=self._plot_settings['plot_output_service'] == 'mapbox')
            choro.update(regds['manager'])
            self._figure.add_trace(choro)

    def plot_grids(self, remove_underlying: bool = False):
        """Plots the grid datasets within the builder.

        Merges all of the datasets together, and plots it as a single plot trace.
        """
        merged = pd.concat(self.get_query_data('grids'))
        merged.drop_duplicates(inplace=True)

        if remove_underlying:
            try:
                merged = self._remove_underlying_grid(self.get_main()['data'], merged)
            except KeyError:
                raise ValueError("Can not remove underlying grid when there is no main dataset.")

        merged['text'] = 'GRID'
        choro = _prepare_choropleth_trace(merged, mapbox=self._plot_settings['plot_output_service'] == 'mapbox')
        choro.update(text=merged['text'])
        choro.update(self._grid_manager)

        self._figure.add_trace(choro)

    def plot_dataset(self):
        """Plots the main dataset within the builder.

        If qualitative, the dataset is split into uniquely labelled plot traces.
        """

        dataset = self['main']

        if dataset is None:
            raise ValueError("There is no main dataset to plot.")
        df = dataset['data']

        # qualitative dataset
        if dataset['VTYPE'] == 'str':
            df['text'] = 'BEST OPTION: ' + df['value_field']
            colorscale = dataset['manager'].get('colorscale')
            try:
                colorscale = tryGetScale(colorscale)
            except AttributeError:
                pass

            # we need to get the colorscale information somehow.
            sep = {}
            mapbox = self._plot_settings['plot_output_service'] == 'mapbox'

            df['temp_value'] = df['value_field']
            df['value_field'] = 0

            for i, name in enumerate(df['temp_value'].unique()):
                sep[name] = df[df['temp_value'] == name]

            # TODO: we need to fix this (qualitative data set plotting)
            manager = deepcopy(dataset['manager'])
            if isinstance(colorscale, dict):
                for k, v in sep.items():
                    choro = _prepare_choropleth_trace(v, mapbox=mapbox)
                    try:
                        manager['colorscale'] = solid_scale(colorscale[k])
                    except KeyError:
                        raise ValueError("If the colorscale is a map, you must provide hues for each option.")
                    choro.update(manager)
                    self._figure.add_trace(choro)
            elif isinstance(colorscale, list) or isinstance(colorscale, tuple):
                for i, v in enumerate(sep.values()):
                    print('ITEM', i)
                    choro = _prepare_choropleth_trace(v, mapbox=mapbox).update(colorscale=solid_scale(colorscale[i]))
                    self._figure.add_trace(choro)
            else:
                raise ValueError("There was an error reading the colorscale.")

        # quantitative dataset
        else:
            df['text'] = 'VALUE: ' + df['value_field'].astype(str)
            choro = _prepare_choropleth_trace(df,
                                              mapbox=self._plot_settings['plot_output_service'] == 'mapbox')
            choro.update(text=df['text'])
            choro.update(dataset['manager'])
            self._figure.add_trace(choro)

    def plot_outlines(self):
        """Plots the outline datasets within the builder.

        All of the outlines are treated as separate plot traces.
        The datasets must first be converted into point-like geometries.
        """
        for outname, outds in self.get_outlines().items():
            try:
                outds['data'] = gcg.pointify_geodataframe(outds['data'], keep_geoms=False)
            except KeyError:
                pass
            print(outds['data'])
            scatt = _prepare_scattergeo_trace(outds['data'], separate=True, disjoint=True,
                                              mapbox=self._plot_settings['plot_output_service'] == 'mapbox')
            scatt.update(outds['manager'])
            self._figure.add_trace(scatt)

    def plot_points(self):
        """Plots the point datasets within the builder.

        All of the point are treated as separate plot traces.
        """
        for poiname, poids in self.get_points().items():
            scatt = _prepare_scattergeo_trace(poids['data'], separate=False, disjoint=False,
                                              mapbox=self._plot_settings['plot_output_service'] == 'mapbox')
            scatt.update(poids['manager'])
            self._figure.add_trace(scatt)

    def set_mapbox(self, accesstoken: str):
        self._plot_settings['plot_output_service'] = 'mapbox'
        self._figure.update_layout(mapbox_accesstoken=accesstoken)

    def build_plot(self):
        """Builds the final plot by adding traces in order.

        Invokes the functions in the following order:
        1) plot regions
        2) plot grids
        3) plot dataset
        4) plot outlines
        5) plot points

        In the future we should alter these functions to
        allow trace order implementation.
        """
        try:
            self.plot_regions()
        except ValueError:
            pass
        try:
            self.plot_grids(remove_underlying=True)
        except ValueError:
            pass
        try:
            self.plot_dataset()
        except ValueError:
            pass
        try:
            self.plot_outlines()
        except ValueError:
            pass
        try:
            self.plot_points()
        except ValueError:
            pass

    def output_figure(self, filepath: str, **kwargs):
        print()

    def display_figure(self, **kwargs):
        print()

    def reset(self):
        print()


if __name__ == '__main__':
    x = DataFrame(dict(lats=[10, 10, 20, 20, 30, 30], lons=[10, 10, 20, 20, 30, 30]))

    pb = PlotBuilder()
    pb.set_main(x, latitude_field='lats', longitude_field='lons', binning_fn=lambda x: 10,
                manager=dict(colorscale='Viridis'))

    pb.add_region('CCA1', 'CANADA')
    pb.add_region('CCA2', 'FRANCE')
    y = GeoDataFrame(
        geometry=[Polygon([[-45.70, -32.99], [70.49, -32.99], [70.49, 62.51], [-45.70, 62.51], [-45.70, -32.99]])],
        crs='EPSG:4326')

    pb.add_region('TESTER', y)
    pb.add_region('TESTER2', y.copy(deep=True))
    # pb.add_region('tony', data=x.copy(deep=True), latitude_field='lats', longitude_field='lons')
    pb.add_grid('tony', data=x.copy(deep=True), latitude_field='lats', longitude_field='lons', binning_fn=lambda x: 10)

    # pb.searchv2('regions:-[CCA1+CA2+CCA3]')
    pb.logify_scale('main')
    pb.clip_datasets('main', 'regions', op='contains')

    # pb.auto_focus(on='regions')
    # pb.logify_scale()
    # pb.discretize_scale()
    # pb.auto_focus()
    # print('FINAL', pb['main'])
    # pb.remove_empties('main', add_to_plot=True)
    # pb.logify_scale('grids')
    # pb.auto_grid('main', by_bounds=True, hex_resolution=3)
    # print(pb.get_grids())
    # print(pb.get_grids())
    # print('RESULT', pb['grid:tony']['data']['value_field'])
    # print(pb['main']['main']['MAIN']['manager'])

    # dataset = _read_dataset(x, latitude_field='lats', longitude_field='lons', binning_fn=dict(fn=sum))
    # _hexbinify_dataset(dataset, 3)
    # print(dataset['data']['value_field'])
