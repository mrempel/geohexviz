from copy import deepcopy
from typing import Any, Tuple, ClassVar, Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from os import path
from os.path import join as pjoin

import plotly.colors
from plotly.graph_objs import Figure, Choropleth, Scattergeo, Choroplethmapbox, Scattermapbox
from shapely.geometry import Point, Polygon

from geoviz.utils.util import fix_filepath, get_sorted_occurrences, generate_dataframe_random_ids, get_column_type, \
    simplify_dicts, dict_deep_update, get_percdiff
from geoviz.utils import geoutils as gcg
from geoviz.utils import plot_util as butil

from geoviz.utils.colorscales import solid_scale, configureColorWithAlpha, configureScaleWithAlpha, getDiscreteScale, \
    tryGetScale

from geopandas import GeoDataFrame
from pandas import DataFrame
from collections import defaultdict

plotly.io.kaleido.scope.default_format = 'pdf'

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

StrDict = Dict[str, Any]


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

Assumptions:

Some of the functions will be limited to only the main dataset, and others will have to use query functions.

Notes for Monday:

- Help Nicholi produce his visualization
- Optimize imports, functions, etc...
- Test edge cases with clipping, then plotting or invoking other functions.

Notes for Tuesday:

- Meet with Mark and discuss progress track and such
- Alter query functions, for both internal and external usage

Notes for Thursday:
- Continue with testing, documentation
- Write in the reference document
"""


def _validate_dataset(dataset: StrDict):
    """Validates a dataset.

    :param dataset: The dataset to validate
    :type dataset: StrDict
    """
    if 'data' not in dataset:
        raise ValueError("There must be a 'data' member present in the dataset.")


def _read_dataset(dataset: StrDict, default_manager: StrDict = None, allow_manager_updates: bool = True):
    """Converts a dataset into a usable dataset for the builder.

    Converts the dataset into a proper type.

    :param dataset: The dataset to convert
    :type dataset: StrDict
    :param default_manager: The default manager to give to the dataset
    :type default_manager: StrDict
    :param allow_manager_updates: Whether or not to allow overwriting of the default manager
    :type allow_manager_updates: bool
    """

    _validate_dataset(dataset)
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

            dataset['data'] = GeoDataFrame(data, geometry=gpd.points_from_xy(longitude_field, latitude_field,
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


def _update_manager(dataset: StrDict, updates: StrDict = None, overwrite: bool = False, **kwargs):
    updates = simplify_dicts(fields=updates, **kwargs)
    if overwrite:
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

    if vtype == 'unk':
        raise TypeError("The binning field is not a valid type, must be string or numerical column.")

    if binning_fn is None:
        binning_fn = _group_functions['bestworst'] if vtype == 'str' else _group_functions['count']

    return gcg.conform_geogeometry(
        gcg.bin_by_hex(data, binning_fn, *args, binning_field=binning_field, result_name='value_field',
                       add_geoms=True, **kwargs)), vtype


def _hexify_dataset(dataset: StrDict, hex_resolution: int):
    dataset['data'] = _hexify_data(dataset['data'], dataset.pop('hex_resolution', hex_resolution))


def _bin_dataset_by_hex(dataset: StrDict):
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


def _hexbinify_dataset(dataset: StrDict, hex_resolution: int):
    _hexify_dataset(dataset, hex_resolution)
    _bin_dataset_by_hex(dataset)


def _create_dataset(data, fields: StrDict = None, **kwargs) -> StrDict:
    if fields is None:
        fields = {}
    fields.update(dict(data=data, **kwargs))
    return fields


def _split_name(name: str) -> Tuple[str, str]:
    lind = name.index(':')
    return name[:lind], name[lind + 1:]


# this function should standalone.
def _prepare_general_dataset(dataset: StrDict, **kwargs):
    try:
        dataset['data'] = butil.get_shapes_from_world(dataset['data'])
    except (KeyError, ValueError, TypeError):
        # logger.debug("If name was a country or continent, the process failed.")
        pass

    _read_dataset(dataset, **kwargs)
    dataset['data'] = gcg.conform_geogeometry(dataset['data'], fix_polys=True)[['geometry']]
    # logger.debug('dataframe geometry conformed to GeoJSON standard.')

    if dataset.pop('to_boundary', False):
        dataset['data'] = gcg.unify_geodataframe(dataset['data'])
    dataset['data']['value_field'] = 0
    dataset['VTYPE'] = 'num'


def _check_name(name: str):
    if any(not i.isalnum() and i != '_' for i in name):
        raise ValueError("The name can not have any non alphanumeric characters in it besides the underscore.")


class PlotBuilder:
    """This class contains a Builder implementation for visualizing Plotly Hex data.
    """

    # default choropleth manager (includes properties for choropleth only)
    _default_dataset_manager: ClassVar[StrDict] = dict(
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
        hoverinfo='location+z+text'
    )

    # default scatter plot manager (includes properties for scatter plots only)
    _default_point_manager: ClassVar[StrDict] = dict(
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

    _default_grid_manager: ClassVar[StrDict] = dict(
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

    _default_region_manager: ClassVar[StrDict] = dict(
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

    _default_outline_manager: ClassVar[StrDict] = dict(
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
    _default_figure_manager: ClassVar[StrDict] = dict(
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
                r=0, l=0, t=50, b=50
            )
        )
    )

    def __init__(
            self,
            main: StrDict = None,
            regions: Dict[str, StrDict] = None,
            grids: Dict[str, StrDict] = None,
            outlines: Dict[str, StrDict] = None,
            points: Dict[str, StrDict] = None
    ):
        """Initializer for instances of PlotBuilder.

        Initializes a new PlotBuilder with the given main dataset
        alongside any region, grid, outline, point type datasets.

        :param main: The main dataset for this builder
        :type main: StrDict
        :param regions: A set of region-type datasets for this builder
        :type regions: Dict[str, StrDict]
        :param grids: A sed of grid-type datasets for this builder
        :type grids: Dict[str, StrDict]
        :param outlines: A set of outline-type datasets for this builder
        :type outlines: Dict[str, StrDict]
        :param points: A set of point-type datasets for this builder
        :type points: Dict[str, StrDict]
        """

        self._figure = Figure()
        self.update_figure(**self._default_figure_manager)
        self._figure.update_layout(width=800, height=600, autosize=False)

        self._container = {
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        # grids will all reference this manager
        self._grid_manager = deepcopy(self._default_grid_manager)

        self._output_service = 'plotly'
        self.default_hex_resolution = 3

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

    @property
    def plot_output_service(self) -> str:
        """Retrieves the current plot output service for the builder.

        :return: The current plot output service
        :rtype: str
        """
        return self.get_plot_output_service()

    @plot_output_service.setter
    def plot_output_service(self, service: str):
        """Sets the plot output service for this builder.

        :param service: The output service (one of 'plotly', 'mapbox')
        :type service: str
        """
        self.set_plot_output_service(service)

    def get_plot_output_service(self) -> str:
        """Retrieves the current plot output service for the builder.

        :return: The current plot output service
        :rtype: str
        """
        return self._output_service

    def set_plot_output_service(self, service: str):
        """Sets the plot output service for this builder.

        :param service: The output service (one of 'plotly', 'mapbox')
        :type service: str
        """
        if service not in ['plotly', 'mapbox']:
            raise ValueError("The output service must be one of ['plotly', 'mapbox'].")
        self._output_service = service

    def __setitem__(self, key: str, value: StrDict):
        """Overwritten setitem method.

        Does one of the following:
        1) Sets the main dataset
        2) Adds a region dataset
        3) Adds a grid dataset
        4) Adds an outline dataset
        5) Adds a point dataset

        The key must be in the form:
        <main> or <region|grid|outline|point>:<name>

        :param key: The key as shown above
        :type key: str
        :param value: A dict representing a valid dataset
        :type value: StrDict
        """

        if not isinstance(key, str):
            raise KeyError("You may not use a non-string key.")

        if not isinstance(value, dict):
            raise ValueError("The value must be a string dictionary (Dict[str, Any])")

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

        if key == 'main':
            self.remove_main()

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
        _read_dataset(dataset := _create_dataset(data, fields=fields, **kwargs),
                      default_manager=deepcopy(self._default_dataset_manager))
        _hexbinify_dataset(dataset, 3)
        dataset['DSTYPE'] = 'MN'
        self._container['main'] = dataset

    def _get_main(self):
        try:
            return self._container['main']
        except KeyError:
            raise ValueError(f"The main dataset could not be found.")

    def get_main(self):
        return deepcopy(self._get_main())

    def remove_main(self):
        try:
            del self._container['main']
        except KeyError:
            raise ValueError("The main dataset could not be found.")

    def update_main_manager(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        _update_manager(self._get_main(), updates=updates, overwrite=overwrite, **kwargs)

    """
    REGION FUNCTIONS
    """

    def add_region(self, name: str, data, fields: StrDict = None, **kwargs):
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
        _check_name(name)
        _prepare_general_dataset(dataset := _create_dataset(data, fields=fields, **kwargs),
                                 default_manager=deepcopy(self._default_region_manager),
                                 allow_manager_updates=True)
        if 'Point' in dataset['data'].geom_type.values:
            raise NotImplementedError("Can not have region geometry type as points.")
        dataset['DSTYPE'] = 'RGN'
        self._get_regions()[name] = dataset

    def _get_region(self, name: str) -> StrDict:
        """Retrieves a region dataset from the builder.

        Internal version.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        try:
            return self._get_regions()[name]
        except KeyError:
            raise KeyError(f"The region-type dataset with the name '{name}' could not be found.")

    def get_region(self, name: str) -> StrDict:
        """Retrieves a region dataset from the builder.

        External version, returns a deepcopy.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        return deepcopy(self._get_region(name))

    def _get_regions(self) -> Dict[str, StrDict]:
        """Retrieves the region datasets from the builder.

        Internal version.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return self._container['regions']

    def get_regions(self) -> Dict[str, StrDict]:
        """Retrieves the region datasets from the builder.

        External version, returns a deepcopy.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_regions())

    def remove_region(self, name: str, pop: bool = False) -> StrDict:
        """Removes a region dataset from the builder.

        :param name: The name of the dataset to remove
        :type name: str
        :param pop: Whether to return the removed dataset or not
        :type pop: bool
        :return: The removed dataset (pop=True)
        :rtype: StrDict
        """
        try:
            region = self._get_regions().pop(name)
            if pop:
                return region
        except KeyError:
            raise ValueError(f"The region-type dataset with the name '{name}' could not be found.")

    def update_region_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a region or regions.

        The manager consists of Plotly properties.
        If the given name is none, all region datasets will be updated.

        :param name: The name of the dataset to update
        :type name: str
        :param updates: A dict containing updates for the dataset(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the dataset(s)
        :type kwargs: **kwargs
        """

        if name is None:
            for _, v in self._get_regions().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_region(name), updates=updates, overwrite=overwrite, **kwargs)

    def reset_regions(self):
        """Resets the regions within the builder to empty.
        """
        self._container['regions'] = {}

    """
    GRID FUNCTIONS
    """

    def add_grid(self, name: str, data, fields: StrDict = None, **kwargs):
        """Adds a grid-type dataset to the builder.

        Grid-type datasets should consist of Polygon-like or Point-like geometries.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param fields: Additional information for this dataset
        :type fields: Dict[str, Any]
        :param kwargs: Additional fields for this dataset
        :type kwargs: **kwargs
        """
        _check_name(name)
        _prepare_general_dataset(dataset := _create_dataset(data, fields=fields, **kwargs),
                                 default_manager=self._grid_manager, allow_manager_updates=False)
        _hexbinify_dataset(dataset, 3)  # change to default hex res
        dataset['DSTYPE'] = 'GRD'
        self._get_grids()[name] = dataset

    def _get_grid(self, name: str) -> StrDict:
        """Retrieves a grid dataset from the builder.

        Internal version.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        try:
            return self._get_grids()[name]
        except KeyError:
            raise KeyError(f"The grid-type dataset with the name '{name}' could not be found.")

    def get_grid(self, name: str) -> StrDict:
        """Retrieves a grid dataset from the builder.

        External version, returns a deepcopy.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        return deepcopy(self._get_grid(name))

    def _get_grids(self) -> Dict[str, StrDict]:
        """Retrieves the grid datasets from the builder.

        Internal version.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return self._container['grids']

    def get_grids(self) -> Dict[str, StrDict]:
        """Retrieves the grid datasets from the builder.

        External version, returns a deepcopy.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_grids())

    def remove_grid(self, name: str, pop: bool = False) -> StrDict:
        """Removes a grid dataset from the builder.

        :param name: The name of the dataset to remove
        :type name:
        :param pop: Whether to return the removed dataset or not
        :type pop: bool
        :return: The removed dataset (pop=True)
        :rtype: StrDict
        """
        try:
            grids = self._get_grids()
            grid = grids.pop(name)

            if len(grids) == 0:
                self.reset_grids()

            if pop:
                return grid
        except KeyError:
            raise ValueError(f"The grid-type dataset with the name '{name}' could not be found.")

    def update_grid_manager(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the general grid manager.

        :param updates: A dict of updates for the manager
        :type updates: StrDict
        :param overwrite: Whether or not to override existing manager properties
        :type overwrite: bool
        :param kwargs: Any additional updates for the manager
        :type kwargs: **kwargs
        """
        dct = {'manager': self._grid_manager}
        _update_manager(dct, updates=updates, overwrite=overwrite, **kwargs)
        self._grid_manager = dct['manager']

    def reset_grids(self):
        """Resets the grid dataset container to it's original state.
        """
        self._container['grids'] = {}
        self._grid_manager = deepcopy(self._default_grid_manager)  # reset to default

    """
    OUTLINE FUNCTIONS
    """

    def add_outline(self, name: str, data, fields: StrDict = None, **kwargs):
        """Adds a outline-type dataset to the builder.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param fields: Additional information for this dataset
        :type fields: Dict[str, Any]
        :param kwargs: Additional fields for this dataset
        :type kwargs: **kwargs
        """
        _check_name(name)
        _prepare_general_dataset(dataset := _create_dataset(data, fields=fields, **kwargs),
                                 default_manager=deepcopy(self._default_outline_manager),
                                 allow_manager_updates=True)
        # consider the pros and cons to converting the dataset to points here
        dataset['DSTYPE'] = 'OUT'
        self.get_outlines()[name] = dataset

    def _get_outline(self, name: str) -> StrDict:
        """Retrieves a outline dataset from the builder.

        Internal version.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        try:
            return self._get_outlines()[name]
        except KeyError:
            raise KeyError(f"The outline-type dataset with the name '{name}' could not be found.")

    def get_outline(self, name: str) -> StrDict:
        """Retrieves a outline dataset from the builder.

        External version, returns a deepcopy.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        return deepcopy(self._get_outline(name))

    def _get_outlines(self) -> Dict[str, StrDict]:
        """Retrieves the outline datasets from the builder.

        Internal version.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return self._container['outlines']

    def get_outlines(self) -> Dict[str, StrDict]:
        """Retrieves the outline datasets from the builder.

        External version, returns a deepcopy.

        :return: The retrieved datasets
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_outlines())

    def remove_outline(self, name: str, pop: bool = False) -> StrDict:
        """Removes an outline dataset from the builder.

        :param name: The name of the dataset to remove
        :type name:
        :param pop: Whether to return the removed dataset or not
        :type pop: bool
        :return: The removed dataset (pop=True)
        :rtype: StrDict
        """
        try:
            outline = self._get_outlines().pop(name)
            if pop:
                return outline
        except KeyError:
            raise ValueError(f"The outline-type dataset with the name '{name}' could not be found.")

    def update_outline_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a outline or outlines.

        The manager consists of Plotly properties.
        If the given name is none, all outline datasets will be updated.

        :param name: The name of the dataset to update
        :type name: str
        :param updates: A dict containing updates for the dataset(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the dataset(s)
        :type kwargs: **kwargs
        """

        if name is None:
            for _, v in self._get_outlines().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_outline(name), updates=updates, overwrite=overwrite, **kwargs)

    def reset_outlines(self):
        """Resets the outlines within the builder to empty.
        """
        self._container['outlines'] = {}

    """
    POINT FUNCTIONS
    """

    def add_point(self, name: str, data, fields: StrDict = None, **kwargs):
        """Adds a outline-type dataset to the builder.

        Ideally the dataset's 'data' member should contain
        lat/long columns or point like geometry column. If the geometry column
        is present and contains no point like geometry, the geometry will be converted
        into a bunch of points.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param fields: Additional information for this dataset
        :type fields: Dict[str, Any]
        :param kwargs: Additional fields for this dataset
        :type kwargs: **kwargs
        """

        _check_name(name)
        _prepare_general_dataset(dataset := _create_dataset(data, fields=fields, **kwargs),
                                 default_manager=deepcopy(self._default_point_manager),
                                 allow_manager_updates=True)
        dataset['data'] = gcg.pointify_geodataframe(dataset['data'])
        dataset['DSTYPE'] = 'PNT'
        self._get_points()[name] = dataset

    def _get_point(self, name: str) -> StrDict:
        """Retrieves a point dataset from the builder.

        Internal version.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        try:
            return self.get_points()[name]
        except KeyError:
            raise KeyError(f"The point-type dataset with the name '{name}' could not be found.")

    def get_point(self, name: str) -> StrDict:
        """Retrieves a point dataset from the builder.

        External version, returns a deepcopy.

        :param name: The name of the dataset
        :type name: str
        :return: The retrieved dataset
        :rtype: StrDict
        """
        return deepcopy(self._get_point(name))

    def _get_points(self) -> Dict[str, StrDict]:
        """Retrieves the collection of point datasets in the builder.

        Internal version.

        :return: The point datasets within the builder
        :rtype: Dict[str, StrDict]
        """
        return self._container['points']

    def get_points(self) -> Dict[str, StrDict]:
        """Retrieves the collection of point datasets in the builder.

        External version, returns a deepcopy.

        :return: The point datasets within the builder
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_points())

    def remove_point(self, name: str, pop: bool = False) -> StrDict:
        """Removes a point dataset from the builder.

        :param name: The name of the dataset to remove
        :type name: str
        :param pop: Whether to return the removed dataset or not
        :type pop: bool
        :return: The removed dataset (pop=True)
        :rtype: StrDict
        """
        try:
            point = self._get_points().pop(name)
            if pop:
                return point
        except KeyError:
            raise ValueError(f"The point-type dataset with the name '{name}' could not be found.")

    def update_point_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a point or points.

        The manager consists of Plotly properties.
        If the given name is none, all point datasets will be updated.

        :param name: The name of the dataset to update
        :type name: str
        :param updates: A dict containing updates for the dataset(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the dataset(s)
        :type kwargs: **kwargs
        """
        if name is None:
            for _, v in self._get_points().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_point(name), updates=updates, overwrite=overwrite, **kwargs)

    def reset_points(self):
        """Resets the point dataset container to its original state.
        """
        self._container['points'] = {}

    """
    FIGURE FUNCTIONS
    """

    def update_figure(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the figure properties on the spot.

        :param updates: A dict of properties to update the figure with
        :type updates: StrDict
        :param overwrite: Whether to overwrite existing figure properties or not
        :type overwrite: bool
        :param kwargs: Any other updates for the figure
        :type kwargs: **kwargs
        """
        updates = simplify_dicts(fields=updates, **kwargs)
        self._figure.update_geos(updates.pop('geos', {}), overwrite=overwrite)
        self._figure.update(overwrite=overwrite, **updates)

    """
    DATA ALTERING FUNCTIONS
    """

    def apply_to_query(self, name: str, fn, *args, allow_empty: bool = True, **kwargs):
        """Applies a function to the datasets within a query.

        For advanced users and not to be used carelessly.
        The functions first argument must be the dataset.

        :param name: The query of the datasets to apply the function to
        :type name: str
        :param fn: The function to apply
        :type fn: Callable
        :param allow_empty: Whether to allow query arguments that retrieved empty results or not
        :type allow_empty: bool
        """

        datasets = self._search(name)

        lst = []

        if not datasets and not allow_empty:
            raise ValueError("The query submitted returned an empty result.")
        if 'data' in datasets:
            lst.append(fn(datasets, *args, **kwargs))
        else:
            for _, v in datasets.items():
                if not v and not allow_empty:
                    raise ValueError("The query submitted returned an empty result.")
                if 'data' in v:
                    lst.append(fn(v, *args, **kwargs))
                else:
                    for _, vv in v.items():
                        if not vv and not allow_empty:
                            raise ValueError("The query submitted returned an empty result.")
                        if 'data' in vv:
                            lst.append(fn(vv, *args, **kwargs))
                        else:
                            for _, vvv in vv.items():
                                if not vvv and not allow_empty:
                                    raise ValueError("The query submitted returned an empty result.")
                                if 'data' in vvv:
                                    lst.append(fn(vvv, *args, **kwargs))
                                else:
                                    raise ValueError("Error when applying function to query.")
        return lst

    def remove_empties(self, empty_symbol: Any = 0, add_to_plot: bool = False):
        """Removes empty entries from the main dataset.

        The empty entries may then be added to the plot as a grid.

        :param empty_symbol: The symbol that constitutes an empty value in the dataset
        :type empty_symbol: Any
        :param add_to_plot: Whether to add the empty cells to the plot or not
        :type add_to_plot: bool
        """

        try:
            dataset = self.get_main()
        except ValueError:
            raise ValueError("There is no main dataset to remove empty entries from.")

        if add_to_plot:
            empties = dataset['data'][dataset['data']['value_field'] == empty_symbol]
            if not empties.empty:
                empties['value_field'] = 0
                self.get_grids()['|*EMPTY*|'] = empties
        dataset['data'] = dataset['data'][dataset['data']['value_field'] != empty_symbol]

    # this is both a data altering, and plot altering function
    # TODO: this function should only apply to the main dataset
    def logify_scale(self, **kwargs):
        """Makes the scale of the main datasets logarithmic.

        This function changes the tick values and tick text of the scale.
        The numerical values on the scale are the exponent of the tick text,
        i.e the text of 1 on the scale actually represents the value of zero,
        and the text of 1000 on the scale actually represents the value of 3.

        :param kwargs: Keyword arguments to be passed into logify functions
        :type kwargs: **kwargs
        """

        try:
            dataset = self.get_main()
        except ValueError:
            raise ValueError("There is no main dataset to convert to a logarithmic scale.")

        if dataset['VTYPE'] == 'str':
            raise TypeError("A qualitative dataset can not be converted into a logarithmic scale.")
        _update_manager(dataset, butil.logify_scale(dataset['data'], **kwargs))

    # TODO: this function should be able to take any two big or small queries.
    def clip_datasets(self, clip: str, to: str, method: str = 'sjoin', operation: str = 'intersects'):
        """Clips a query of datasets to another dataset.

        There are two methods for this clipping:
        1) sjoin -> Uses GeoPandas spatial join in order to clip geometries
                    that (intersect, are within, contain, etc.) the geometries
                    acting as the clip.
        2) gpd  ->  Uses GeoPandas clip function in order to clip geometries
                    to the boundary of the geometries acting as the clip.

        :param clip: The query for the datasets that are to be clipped to another
        :type clip: GeoDataFrame
        :param to: The query for the datasets that are to be used as the boundary
        :type to: GeoDataFrame
        :param method: The method to use when clipping, one of 'sjoin', 'gpd'
        :type method: str
        :param operation: The operation to apply when using sjoin (spatial join operation)
        :type operation: str
        """

        datas = self.apply_to_query(to, lambda dataset: dataset['data'], big_query=True)

        def gpdhelp(dataset: dict):
            dataset['data'] = butil.gpd_clip(dataset['data'], datas)

        def sjoinhelp(dataset: dict):
            dataset['data'] = butil.sjoin_clip(dataset['data'], datas, operation=operation)

        if method == 'gpd':
            self.apply_to_query(clip, gpdhelp, big_query=True)
        elif method == 'sjoin':
            self.apply_to_query(clip, sjoinhelp, big_query=True)
        else:
            raise ValueError("The selected method must be one of ['gpd', 'sjoin'].")

    # check methods of clipping
    def simple_clip(self, method: str = 'sjoin'):
        """Quick general clipping.

        This function clips the main dataset and grid datasets to the region and outline datasets.
        The function also clips the point datasets to the main, region, grid, and outline datasets.

        :param method: The method to use when clipping, one of 'sjoin' or 'gpd'
        :type method: str
        """

        self.clip_datasets('main+grids', 'regions+outlines', method=method, operation='intersects')
        self.clip_datasets('points', 'main+regions+outlines+grids', method=method, operation='within')

    def _remove_underlying_grid(self, df: GeoDataFrame, gdf: GeoDataFrame):
        """Removes the pieces of a GeoDataFrame that lie under another.

        :param df: The overlayed dataframe (after alteration)
        :type df: GeoDataFrame
        :param gdf: The merged underlying dataframe (after alteration)
        :type gdf: GeoDataFrame
        """

        if not df.empty and not gdf.empty:
            gdf = gpd.overlay(gdf, df[['value_field', 'geometry']],
                              how='difference')
            gdf = gcg.remove_other_geometries(gdf, 'Polygon')
            return gdf

    """
    PLOT ALTERING FUNCTIONS
    """

    # TODO: This function should only apply to the main dataset (maybe, think more)
    def opacify_colorscale(self, alpha: float = None):
        """Conforms the opacity of the colorbar of the main dataset to an alpha value.

        The alpha value can be passed in as a parameter, otherwise it is taken
        from the marker.opacity property within the dataset's manager.

        :param alpha: The alpha value to conform the colorscale to
        :type alpha: float
        """

        dataset = self.get_main()
        butil.opacify_colorscale(dataset, alpha=alpha)

    def auto_focus(self, on: str = 'main', center_on: bool = False, rotation_on: bool = True, ranges_on: bool = True,
                   buffer_lat: tuple = (0, 0), buffer_lon: tuple = (0, 0), validate: bool = True):
        """Focuses on dataset(s) within the plot.

        Collects the geometries of the queried datasets in order to
        obtain a boundary to focus on.

        In the future using a GeoSeries may be looked into for cleaner code.

        :param on: The query for the dataset(s) to be focused on
        :type on: str
        :param center_on: Whether or not to add a center component to the focus
        :type center_on: bool
        :param rotation_on: Whether or not to add a projection rotation to the focus
        :type rotation_on: bool
        :param ranges_on: Whether or not to add a lat axis, lon axis ranges to the focus
        :type ranges_on: bool
        :param buffer_lat: A low and high bound to add and subtract from the lataxis range
        :type buffer_lat: Tuple[float, float]
        :param buffer_lon: A low and high bound to add and subtract from the lonaxis range
        :type buffer_lon: Tuple[float, float]
        :param validate: Whether or not to validate the ranges
        :type validate: bool
        """

        geoms = []
        self.apply_to_query(on, lambda ds: geoms.extend(list(ds['data'].geometry)))

        if not geoms:
            raise ValueError("There are no geometries in your query to focus on.")

        geos = {}
        if ranges_on:
            lonrng, latrng = gcg.find_ranges_simple(geoms)
            geos['lataxis_range'] = (latrng[0] - buffer_lat[0], latrng[1] + buffer_lat[1])
            geos['lonaxis_range'] = (lonrng[0] - buffer_lon[0], lonrng[1] + buffer_lon[1])

            if validate:
                latlow, lathigh = geos['lataxis_range']
                lonlow, lonhigh = geos['lonaxis_range']

                lonlowdiff, lonhighdiff = get_percdiff(lonlow, -180), get_percdiff(lonhigh, 180)
                latlowdiff, lathighdiff = get_percdiff(latlow, -90), get_percdiff(lathigh, 90)

                if lonlowdiff <= 5 and lonhighdiff <= 5:
                    geos['lonaxis_range'] = (-180, 180)
                if latlowdiff <= 5 and lathighdiff <= 5:
                    geos['lataxis_range'] = (-90, 90)

        center = gcg.find_center_simple(geoms)
        center = dict(lon=center.x, lat=center.y)

        if rotation_on:
            geos['projection_rotation'] = center

        if center_on:
            geos['center'] = center

        self._figure.update_geos(**geos)

    def auto_grid(self, query: str = 'main', by_bounds: bool = False, hex_resolution: int = 3):
        """Makes a grid over the queried datasets.

        :param query: The query for the datasets to have a grid generated over them
        :type query: str
        :param by_bounds: Whether or not to treat the  geometries as a single boundary
        :type by_bounds: bool
        :param hex_resolution: The hexagonal resolution to use for the auto grid
        :type hex_resolution: int
        """

        fn = gcg.generate_grid_over_hexes if by_bounds else gcg.hexify_geodataframe

        def helper(dataset):
            return fn(dataset['data'], hex_resolution=hex_resolution)

        try:
            grid = GeoDataFrame(pd.concat(list(self.apply_to_query(query, helper))), crs='EPSG:4326')
            if not grid.empty:
                grid['value_field'] = 0
                self.get_grids()['|*AUTO*|'] = {'data': gcg.conform_geogeometry(grid), 'manager': self._grid_manager}
                self.get_grids()['|*AUTO*|']['data'].plot()
            else:
                raise ValueError("There may have been an error when generating auto grid, shapes may span too large "
                                 "of an area.")

        except ValueError as e:
            raise e

    def discretize_scale(self, scale_type: str = 'sequential', **kwargs):
        """Converts the color scale of the dataset(s) to a discrete scale.

        :param scale_type: One of 'sequential', 'discrete' for the type of color scale being used
        :type scale_type: str
        :param kwargs: Keyword arguments to be passed into the discretize functions
        :type kwargs: **kwargs
        """

        try:
            dataset = self.get_main()
        except ValueError:
            raise ValueError("There is no main dataset to discretize the scale of.")

        if dataset['VTYPE'] == 'str':
            raise ValueError(f"You can not discretize a qualitative dataset.")
        else:
            low = dataset['manager'].get('zmin', min(dataset['data']['value_field']))
            high = dataset['manager'].get('zmax', max(dataset['data']['value_field']))
            print(dataset['manager'].get('colorscale'))
            dataset['manager']['colorscale'] = getDiscreteScale(dataset['manager'].get('colorscale'), scale_type,
                                                                low, high, **kwargs)

    """
    RETRIEVAL/SEARCHING FUNCTIONS
    
    get_regions(), etc... could also fall under here.
    """

    def search(self, query: str) -> StrDict:
        """Query the builder for specific dataset(s).

        Each query argument should be formatted like:
        <regions|grids|outlines|points|main|all>
        OR
        <region|grid|outline|point>:<name>

        And each query argument can be separated by the '+' character.

        External version.

        :param query: The identifiers for the datasets being searched for
        :type query: str
        :return: The result of the query
        :rtype: StrDict
        """
        return deepcopy(self._search(query))

    def _search(self, query: str, big_query: bool = True, multi_query: bool = True) -> StrDict:
        """Query the builder for specific dataset(s).

        Internal version, see search().

        :param query: The identifiers for the datasets being searched for
        :type query: str
        :return: The result of the query
        :rtype: StrDict
        """
        sargs = query.split('+')

        if not multi_query:
            return self._single_search(query, big_query=big_query)

        if len(sargs) == 1:
            return self._single_search(sargs[0], big_query=big_query)
        else:
            return {k: self._single_search(k, big_query=big_query) for k in sargs}

    def _single_search(self, query: str, big_query: bool = True) -> StrDict:
        """Query the builder for specific datasets()

        Retrieves a query of a single argument only, see _search().

        :param query: The identifier for the dataset(s) being searched for
        :type query: str
        :param big_query: Whether to allow the query argument to represent a collection of datasets or not
        :type big_query: bool
        :return: The result of the query
        :rtype: StrDict
        """

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
            return self._get_region(name)
        elif typer == 'grid':
            return self._get_grid(name)
        elif typer == 'outline':
            return self._get_outline(name)
        elif typer == 'point':
            return self._get_point(name)
        else:
            raise ValueError("The given dataset type does not exist. Must be one of ['region', 'grid', "
                             "'outline', 'point']. "
                             f"Received {typer}.")

    get_query_data = lambda self, name: self.apply_to_query(name, lambda ds: ds['data'])

    def change_colorbar_sizes(self):
        bottompx = self._figure.layout.margin.b
        toppx = self._figure.layout.margin.t
        heightpx = self._figure.layout.height

        print(bottompx, toppx, heightpx)

        self._figure.update_coloraxes(colorbar_lenmode='pixels', colorbar_yanchor='bottom', colorbar_len=heightpx)
        print(self._figure.layout)

    """
    PLOTTING FUNCTIONS
    """

    def plot_regions(self):
        """Plots the region datasets within the builder.

        All of the regions are treated as separate plot traces.
        """
        # logger.debug('adding regions to plot.')
        mapbox = self.plot_output_service == 'mapbox'
        if not self._get_regions():
            raise ValueError("There are no region-type datasets to plot.")
        for regname, regds in self._get_regions().items():
            choro = _prepare_choropleth_trace(regds['data'],
                                              mapbox=mapbox)
            choro.update(regds['manager'])
            self._figure.add_trace(choro)

    def plot_grids(self, remove_underlying: bool = False):
        """Plots the grid datasets within the builder.

        Merges all of the datasets together, and plots it as a single plot trace.
        """
        if not self._get_grids():
            raise ValueError("There are no grid-type datasets to plot.")

        merged = pd.concat(self.apply_to_query('grids', lambda dataset: dataset['data'])).drop_duplicates()

        if remove_underlying:
            try:
                merged = self._remove_underlying_grid(self._get_main()['data'], merged)
            except KeyError:
                raise ValueError("Can not remove underlying grid when there is no main dataset.")

        merged['text'] = 'GRID'
        choro = _prepare_choropleth_trace(merged, mapbox=self.plot_output_service == 'mapbox').update(
            text=merged['text']).update(self._grid_manager)
        self._figure.add_trace(choro)

    def plot_main(self):
        """Plots the main dataset within the builder.

        If qualitative, the dataset is split into uniquely labelled plot traces.
        """
        dataset = self.get_main()
        df = dataset['data']

        # qualitative dataset
        if dataset['VTYPE'] == 'str':
            df['text'] = 'BEST OPTION: ' + df['value_field']
            colorscale = dataset['manager'].pop('colorscale')
            try:
                colorscale = tryGetScale(colorscale)
            except AttributeError:
                pass

            # we need to get the colorscale information somehow.
            sep = {}
            mapbox = self.plot_output_service == 'mapbox'

            df['temp_value'] = df['value_field']
            df['value_field'] = 0

            for i, name in enumerate(df['temp_value'].unique()):
                sep[name] = df[df['temp_value'] == name]

            # TODO: we need to fix this (qualitative data set plotting)
            manager = deepcopy(dataset['manager'])
            if isinstance(colorscale, dict):
                for k, v in sep.items():
                    try:
                        manager['colorscale'] = solid_scale(colorscale[k])
                    except KeyError:
                        raise TypeError("If the colorscale is a map, you must provide hues for each option.")   # TODO: figure out how to change this error
                    choro = _prepare_choropleth_trace(v, mapbox=mapbox).update(showscale=False, showlegend=True).update(
                        manager)
                    self._figure.add_trace(choro)

            elif isinstance(colorscale, list) or isinstance(colorscale, tuple):
                for i, (k, v) in enumerate(sep.items()):
                    print('ITEM', i)

                    if isinstance(colorscale[i], list) or isinstance(colorscale[i], tuple):
                        manager['colorscale'] = solid_scale(colorscale[i][1])
                    else:
                        manager['colorscale'] = solid_scale(colorscale[i])
                    choro = _prepare_choropleth_trace(v, mapbox=mapbox).update(name=k, showscale=False,
                                                                               showlegend=True).update(manager)
                    self._figure.add_trace(choro)
            else:
                raise TypeError("The colorscale must be a map, iterable, or nested iterables.")
        # quantitative dataset
        else:
            df['text'] = 'VALUE: ' + df['value_field'].astype(str)
            choro = _prepare_choropleth_trace(df,
                                              mapbox=self.plot_output_service == 'mapbox')
            choro.update(text=df['text'])
            choro.update(dataset['manager'])
            print(choro.colorbar)
            self._figure.add_trace(choro)

        # self.change_colorbar_sizes()

    def plot_outlines(self, raise_errors: bool = False):
        """Plots the outline datasets within the builder.

        All of the outlines are treated as separate plot traces.
        The datasets must first be converted into point-like geometries.

        :param raise_errors: Whether or not to throw errors upon reaching empty dataframes
        :type raise_errors: bool
        """
        if not self.get_outlines():
            raise ValueError("There are no outline-type datasets to plot.")

        for outname, outds in self.get_outlines().items():
            outds['data'] = gcg.pointify_geodataframe(outds['data'], keep_geoms=False, raise_errors=raise_errors)
            scatt = _prepare_scattergeo_trace(outds['data'], separate=True, disjoint=True,
                                              mapbox=self.plot_output_service == 'mapbox')
            scatt.update(outds['manager'])
            self._figure.add_trace(scatt)

    def plot_points(self):
        """Plots the point datasets within the builder.

        All of the point are treated as separate plot traces.
        """
        if not self.get_points():
            raise ValueError("There are no point-type datasets to plot.")
        for poiname, poids in self.get_points().items():
            scatt = _prepare_scattergeo_trace(poids['data'], separate=False, disjoint=False,
                                              mapbox=self.plot_output_service == 'mapbox')
            scatt.update(poids['manager'])
            self._figure.add_trace(scatt)

    def set_mapbox(self, accesstoken: str):
        """Prepares the builder for a mapbox output.

        Sets figure.layout.mapbox_accesstoken, and plot_settings output service.

        :param accesstoken: A mapbox access token for the plot
        :type accesstoken: str
        """
        self.plot_output_service = 'mapbox'
        self._figure.update_layout(mapbox_accesstoken=accesstoken)

    def build_plot(self, plot_regions: bool = True, plot_grids: bool = True, plot_main: bool = True,
                   plot_outlines: bool = True, plot_points: bool = True, raise_errors: bool = True):
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
        if plot_regions:
            try:
                self.plot_regions()
            except ValueError as e:
                if raise_errors:
                    raise e
        if plot_grids:
            try:
                self.plot_grids(remove_underlying=True)
            except ValueError as e:
                if raise_errors:
                    raise e
        if plot_main:
            try:
                self.plot_main()
            except ValueError as e:
                if raise_errors:
                    raise e
        if plot_outlines:
            try:
                self.plot_outlines()
            except ValueError as e:
                if raise_errors:
                    raise e
        if plot_points:
            try:
                self.plot_points()
            except ValueError as e:
                if raise_errors:
                    raise e

    def output_figure(self, filepath: str, clear_figure: bool = False, **kwargs):
        self._figure.write_image(filepath, **kwargs)
        if clear_figure:
            self.clear_figure()

    def display_figure(self, clear_figure: bool = False, **kwargs):
        self._figure.show(**kwargs)
        if clear_figure:
            self.clear_figure()

    def clear_figure(self):
        self._figure.data = []

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
