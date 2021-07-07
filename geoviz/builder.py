from copy import deepcopy
from enum import Enum
from typing import Any, Tuple, ClassVar, Dict, Union, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from os import path
from os.path import join as pjoin

import plotly.colors
from plotly.graph_objs import Figure, Choropleth, Scattergeo, Choroplethmapbox, Scattermapbox
from shapely.geometry import Point, Polygon

from geoviz.utils.util import fix_filepath, get_sorted_occ, generate_dataframe_random_ids, get_column_type, \
    simplify_dicts, dict_deep_update, get_percdiff
from geoviz.utils import geoutils as gcg
from geoviz.utils import plot_util as butil

from geoviz.utils.colorscales import solid_scale, configure_color_opacity, configureScaleWithAlpha, discretize_cscale, \
    get_scale

from geopandas import GeoDataFrame
from pandas import DataFrame
from collections import defaultdict
import geoviz.errors as gce

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
    'bestworst': get_sorted_occ
}

StrDict = Dict[str, Any]
DFType = Union[str, DataFrame, GeoDataFrame]


def _reset_to_odata(dataset: StrDict):
    """Resets the odata parameter of a dataset.

    :param dataset: The dataset to reset
    :type dataset: StrDict
    """
    dataset['data'] = dataset['odata'].copy(deep=True)


def _prepare_choropleth_trace(gdf: GeoDataFrame, mapbox: bool = False) -> Union[Choropleth, Choroplethmapbox]:
    """Prepares a choropleth trace for a geodataframe.

    :param gdf: The geodataframe to generate a choropleth trace for
    :type gdf: GeoDataFrame
    :param mapbox: Whether to return a mapbox trace or not
    :type mapbox: bool
    :return: The graph trace
    :rtype: Union[Choropleth, Choroplethmapbox]
    """
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


def _prepare_scattergeo_trace(gdf: GeoDataFrame, separate: bool = True, disjoint: bool = False, mapbox: bool = False) -> \
        Union[Scattergeo, Scattermapbox]:
    """Prepares a scattergeo trace for a geodataframe.

    :param gdf: The geodataframe to make a trace for
    :type gdf: GeoDataFrame
    :param separate: Whether to geometries within the geodataframe as separate or not
    :type separate: bool
    :param disjoint: Whether to add np.nan in between entries (plotly recognizes this as separate) or not
    :type disjoint: bool
    :param mapbox: Whether to return a mapbox trace or not
    :type mapbox: bool
    :return: The plotly graph trace
    :rtype: Union[Scattergeo, Scattermapbox]
    """
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


def builder_from_dict(builder_dict: StrDict = None, **kwargs):
    """Makes a builder from a dictionary.

    :param builder_dict: The dictionary to build from
    :type builder_dict: StrDict
    :param kwargs: Keyword arguments for the builder
    :type kwargs: **kwargs
    """
    settings = {}
    if builder_dict:
        settings.update(builder_dict)
    settings.update(kwargs)

    plotly_managers = settings.pop('builder_managers', {})

    builder = PlotBuilder(**settings)
    builder.update_main_manager(plotly_managers.get('main_dataset', {}))
    builder.update_grid_manager(plotly_managers.get('grids', {}))
    builder.update_figure(plotly_managers.get('figure', {}))

    for k, v in plotly_managers.get('regions', {}):
        builder.update_region_manager(v, name=k)
    for k, v in plotly_managers.get('outlines', {}):
        builder.update_outline_manager(v, name=k)
    for k, v in plotly_managers.get('points', {}):
        builder.update_point_manager(v, name=k)

    return builder


def _validate_dataset(dataset: StrDict):
    """Validates a dataset.

    :param dataset: The dataset to validate
    :type dataset: StrDict
    """
    if 'data' not in dataset:
        raise ValueError("There must be a 'data' member present in the dataset.")


def _set_manager(dataset: StrDict, default_manager: StrDict = None, allow_manager_updates: bool = True):
    """Sets the manager of a dataset at the beginning.

    :param dataset: The dataset whose manager to set
    :type dataset: StrDict
    :param default_manager: The default manager for this dataset
    :type default_manager: StrDict
    :param allow_manager_updates: Whether to allow the default manager to be updated for individual datasets or not
    :type allow_manager_updates: bool
    """
    if default_manager is None:
        default_manager = {}
    input_manager = dataset.pop('manager', {})
    dataset['manager'] = {}
    _update_manager(dataset, **default_manager)
    if allow_manager_updates:
        _update_manager(dataset, **input_manager)
    elif input_manager:
        raise ValueError("This dataset may not have a custom manager.")


def _read_data_file(data: str) -> DataFrame:
    """Reads data from a file, based on extension.

    If the file extension is unknown the file is passed
    directly into geopandas.read_file().

    This function uses both geopandas and pandas to read data.

    In the future it may be beneficial to allow the reading
    of databases, and feather.

    :param data: The data to be read.
    :type data: str
    :return: The read data
    :rtype: DataFrame
    """
    try:
        filepath, extension = path.splitext(pjoin(path.dirname(__file__), data))
        filepath = fix_filepath(filepath, add_ext=extension)

        try:
            data = _extension_mapping[extension](filepath)
        except KeyError:
            pass
    except Exception as e:
        raise gce.DataFileReadError(str(e))
        # logger.warning("The general file formats accepted by this application are (.csv, .shp). Be careful.")

    return data


def _read_data(data: DFType, allow_builtin: bool = False) -> GeoDataFrame:
    """Reads the data into a usable type for the builder.

    :param data: The data to be read
    :type data: DFType
    :param allow_builtin: Whether to allow builtin data types or not (countries, continents)
    :type allow_builtin: bool
    :return: A proper geodataframe from the input data
    :rtype: GeoDataFrame
    """
    err_msg = "The data must be a valid filepath, DataFrame, or GeoDataFrame."
    rtype = 'frame'
    try:
        data, rtype = _read_data_file(data), 'file'
    except gce.DataFileReadError:
        if allow_builtin:
            try:
                data, rtype = butil.get_shapes_from_world(data), 'builtin'
            except (KeyError, ValueError, TypeError):
                err_msg = "The data must be a valid country or continent name, filepath, DataFrame, or GeoDataFrame."

    if isinstance(data, DataFrame):
        data = GeoDataFrame(data)
        data['value_field'] = 0
        data.RTYPE = rtype
        data.VTYPE = 'NUM'
    else:
        raise gce.DataReadError(err_msg)

    return data


def _convert_latlong_data(data: GeoDataFrame, latitude_field: str = None, longitude_field: str = None) -> GeoDataFrame:
    """Converts lat/long columns into a proper geometry column, if present.

    :param data: The data that may or may not contain lat/long columns
    :type data: GeoDataFrame
    :param latitude_field: The latitude column within the dataframe
    :type latitude_field: str
    :param longitude_field: The longitude column within the dataframe
    :type longitude_field: str
    :return: The converted dataframe
    :rtype: GeoDataFrame
    """
    if data.empty:
        raise ValueError("If the data passed is a DataFrame, it must not be empty.")

    data = data.copy(deep=True)

    if 'geometry' not in data.columns:

        try:
            latitude_field = data[latitude_field]
        except KeyError:
            if 'latitude' in data.columns:
                latitude_field = data['latitude']
            else:
                raise ValueError(
                    "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                    "and longitude_field entries. Missing latitude_field member.")

        try:
            longitude_field = data[longitude_field]
        except KeyError:
            if 'longitude' in data.columns:
                longitude_field = data['longitude']
            else:
                raise ValueError(
                    "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                    "and longitude_field entries. Missing longitude_field member.")
        data = GeoDataFrame(data, geometry=gpd.points_from_xy(longitude_field, latitude_field,
                                                              crs='EPSG:4326'))
    data.vtype = 'NUM'
    return data


def _convert_latlong_data2(data: GeoDataFrame, latitude_field: str = None, longitude_field: str = None) -> GeoDataFrame:
    """Converts lat/long columns into a proper geometry column, if present.

    :param data: The data that may or may not contain lat/long columns
    :type data: GeoDataFrame
    :param latitude_field: The latitude column within the dataframe
    :type latitude_field: str
    :param longitude_field: The longitude column within the dataframe
    :type longitude_field: str
    :return: The converted dataframe
    :rtype: GeoDataFrame
    """
    if data.empty:
        raise ValueError("If the data passed is a DataFrame, it must not be empty.")

    data = data.copy(deep=True)

    try:
        latitude_field = data[latitude_field]
    except KeyError:
        try:
            latitude_field = data['latitude']
        except KeyError:
            if 'geometry' not in data.columns:
                raise ValueError(
                    "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                    "and longitude_field entries. Missing latitude_field member.")

    try:
        longitude_field = data[longitude_field]
    except KeyError:
        try:
            longitude_field = data['longitude']
        except KeyError:
            if 'geometry' not in data.columns:
                raise ValueError(
                    "If a GeoDataFrame that does not have geometry is passed, there must be latitude_field, "
                    "and longitude_field entries. Missing longitude_field member.")

    try:
        data = GeoDataFrame(data, geometry=gpd.points_from_xy(longitude_field, latitude_field, crs='EPSG:4326'))
    except (TypeError, ValueError):
        pass
    gcg.conform_geogeometry(data)
    data.vtype = 'NUM'
    return data


def _convert_to_hexbin_data(data: GeoDataFrame, hex_resolution: int, binning_args=None,
                            binning_field: str = None, binning_fn: Callable = None, **kwargs) -> GeoDataFrame:
    """Converts a geodataframe into a hexagon-ally binned dataframe.

    :param data: The data to be converted
    :type data: GeoDataFrame
    :param hex_resolution: The hexagonal resolution to use
    :type hex_resolution: int
    :param binning_args: Arguments for the binning functions
    :type binning_args: Iterable
    :param binning_field: The binning column to apply the function on
    :type binning_field: str
    :param binning_fn: The function to apply
    :type binning_fn: Callable
    :param kwargs: Keyword arguments for the function
    :type kwargs: **kwargs
    :return: The hexbinified dataframe
    :rtype: GeoDataFrame
    """
    data = _hexify_data(data, hex_resolution)

    try:
        binning_fn = _group_functions[str(binning_fn)]
    except KeyError:
        pass

    vtype = 'NUM' if binning_field is None else get_column_type(data, binning_field)

    if vtype == 'UNK':
        raise gce.BinValueTypeError("The binning field is not a valid type, must be string or numerical column.")

    if binning_fn is None:
        binning_fn = _group_functions['bestworst'] if vtype == 'STR' else _group_functions['count']

    data = gcg.bin_by_hexid(data, binning_field=binning_field, binning_fn=binning_fn, binning_args=binning_args,
                            result_name='value_field', add_geoms=True, **kwargs)
    vtype = get_column_type(data, 'value_field')
    if vtype == 'UNK':
        raise gce.BinValueTypeError("The result of the binning operation is a column of invalid type. Fatal Error.")
    gcg.conform_geogeometry(data)
    data.VTYPE = vtype
    return data


def _update_manager(dataset: StrDict, updates: StrDict = None, overwrite: bool = False, **kwargs):
    """Updates the manager of a given dataset.

    :param dataset: The dataset whose manager to update
    :type dataset: StrDict
    :param updates: A dict of updates for the manager
    :type updates: StrDict
    :param overwrite: Whether to overwrite the existing manager with the new one or not
    :type overwrite: bool
    :param kwargs: Extra updates for the manager
    :type kwargs: **kwargs
    """
    updates = simplify_dicts(fields=updates, **kwargs)
    if overwrite:
        dataset['manager'] = updates
    else:
        dict_deep_update(dataset['manager'], updates)


def _hexify_data(data: Union[DataFrame, GeoDataFrame], hex_resolution: int) -> Union[DataFrame, GeoDataFrame]:
    """Wrapper for hexifying a geodataframe

    :param data: The geodataframe to hexify
    :type data: Union[DataFrame, GeoDataFrame]
    :param hex_resolution: The hexagonal resolution to use
    :type hex_resolution: int
    :return: The hexified geodataframe
    :rtype: Union[DataFrame, GeoDataFrame]
    """
    return gcg.hexify_dataframe(data, hex_resolution=hex_resolution, add_geom=False, keep_geom=False, as_index=True)


def _bin_by_hex_helper(data: Union[DataFrame, GeoDataFrame], binning_field: str = None, binning_fn=None) -> Tuple[
    Callable, str]:
    """Determines the binning function and preliminary value type for a dataframe which is to be binned.

    :param data: The data which is to be binned
    :type data: Union[DataFrame, GeoDataFrame]
    :param binning_field: The binning field for the data
    :type binning_field: str
    :param binning_fn: The function which is to be performed on the data
    :type binning_fn:
    :return:
    :rtype:
    """
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

    return binning_fn, vtype


def _bin_by_hex(data, *args, binning_field: str = None, binning_fn: Callable = None, **kwargs) -> Tuple[
    GeoDataFrame, str]:
    """Wrapper for binning hexagonal data, and adding geometry to it.

    :param data: The data to be hexagonally binned
    :type data: Union[DataFrame, GeoDataFrame]
    :param args: Binning function arguments
    :type args: *args
    :param binning_field: The binning field of the data
    :type binning_field: str
    :param binning_fn: The function to be performed on the data while binning
    :type binning_fn: Callable
    :param kwargs: Keyword arguments for the binning function
    :type kwargs: **kwargs
    :return: The resulting geodataframe and the value type
    :rtype: Tuple[GeoDataFrame, str]
    """
    binning_fn, vtype = _bin_by_hex_helper(data, binning_field=binning_field, binning_fn=binning_fn)

    result = gcg.bin_by_hexid(data, binning_field=binning_field, binning_fn=binning_fn, result_name='value_field',
                              add_geoms=True, **kwargs)
    vtype = get_column_type(result, 'value_field')
    return result, vtype


def _hexify_dataset(dataset: StrDict, hex_resolution: int):
    """Wrapper for hexaifying a dataset.

    :param dataset: The dataset to be hexified
    :type dataset: StrDict
    :param hex_resolution: The hexagonal resolution to be used
    :type hex_resolution: int
    """
    hres = dataset.get('hex_resolution', None)
    hexbin_info = dataset.get('hexbin_info')
    if hexbin_info:
        if 'resolution' in hexbin_info:
            if hres:
                raise ValueError("Received multiple arguments for hex resolution.")
            hres = hexbin_info['resolution']
    hres = hres if hres is not None else hex_resolution
    dataset['hex_resolution'] = hres
    dataset['data'] = _hexify_data(dataset['data'], hres)


def _bin_dataset_by_hex(dataset: StrDict):
    """Wrapper for hexagonally binning a dataset.

    :param dataset: The dataset to be hexagonally binned
    :type dataset: StrDict
    """
    binning_fn = dataset.pop('binning_fn', None)
    binning_args = dataset.pop('binning_fn_args', ())
    binning_kw = dataset.pop('binning_fn_kwargs', {})

    if isinstance(binning_fn, dict):
        binning_args = binning_fn.get('args', binning_args)
        binning_kw = binning_fn.get('kwargs', binning_kw)
        binning_fn = binning_fn.get('fn', None)

    dataset['data'], dataset['VTYPE'] = _bin_by_hex(dataset['data'], *binning_args,
                                                    binning_field=dataset.pop('binning_field', None),
                                                    binning_fn=binning_fn,
                                                    **binning_kw)


def _split_name(name: str) -> Tuple[str, str]:
    lind = name.index(':')
    return name[:lind], name[lind + 1:]


def _check_name(name: str):
    if any(not i.isalnum() and i != '_' for i in name):
        raise gce.DatasetNamingError("NERR001 - Non-alphanumeric found (besides underscores)")


class PlotStatus(Enum):
    DATA_PRESENT = 0
    NO_DATA = 1


class PlotBuilder:
    """This class contains a Builder implementation for visualizing Plotly Hex data.
    """

    # default choropleth manager (includes properties for choropleth only)
    _default_quantitative_dataset_manager: ClassVar[StrDict] = dict(
        colorscale='Viridis',
        colorbar=dict(title='COUNT'),
        marker=dict(line=dict(color='white', width=0.60), opacity=0.8),
        hoverinfo='location+z+text'
    )

    _default_qualitative_dataset_manager: ClassVar[StrDict] = dict(
        colorscale='Set3',
        colorbar=dict(title='COUNT'),
        marker=dict(line=dict(color='white', width=0.60), opacity=0.8),
        hoverinfo='location+z+text'
    )

    # default scatter plot manager (includes properties for scatter plots only)
    _default_point_manager: ClassVar[StrDict] = dict(
        mode='markers+text',
        marker=dict(
            line=dict(color='black', width=0.3),
            color='white',
            symbol='circle-dot',
            size=6
        ),
        showlegend=False,
        textposition='top center',
        textfont=dict(color='Black', size=5)
    )

    _default_grid_manager: ClassVar[StrDict] = dict(
        colorscale=[[0, 'white'], [1, 'white']],
        zmax=1,
        zmin=0,
        marker=dict(line=dict(color='black'), opacity=0.2),
        showlegend=False,
        showscale=False,
        hoverinfo='text',
        legendgroup='grids'
    )

    _default_region_manager: ClassVar[StrDict] = dict(
        colorscale=[[0, 'rgba(255,255,255,0.525)'], [1, 'rgba(255,255,255,0.525)']],
        marker=dict(line=dict(color="rgba(0,0,0,1)", width=0.65)),
        legendgroup='regions',
        zmin=0,
        zmax=1,
        showlegend=False,
        showscale=False,
        hoverinfo='text'
    )

    _default_outline_manager: ClassVar[StrDict] = dict(
        mode='lines',
        line=dict(color="black", width=1, dash='dash'),
        legendgroup='outlines',
        showlegend=False,
        hoverinfo='text'
    )

    # contains the default properties for the figure
    _default_figure_manager: ClassVar[StrDict] = dict(
        geos=dict(
            projection=dict(type='orthographic'),
            showcoastlines=False,
            showland=True,
            landcolor="rgba(166,166,166,0.625)",
            showocean=True,
            oceancolor="rgb(222,222,222)",
            showlakes=False,
            showrivers=False,
            showcountries=False,
            lataxis=dict(showgrid=True),
            lonaxis=dict(showgrid=True)
        ),
        layout=dict(
            title=dict(text='', x=0.5),
            margin=dict(r=0, l=0, t=0, b=0),
            width=800,
            height=600,
            autosize=False
        )
    )

    def __init__(
            self,
            main_dataset: StrDict = None,
            regions: Dict[str, StrDict] = None,
            grids: Dict[str, StrDict] = None,
            outlines: Dict[str, StrDict] = None,
            points: Dict[str, StrDict] = None,
            use_default_managers: bool = True
    ):
        """Initializer for instances of PlotBuilder.

        Initializes a new PlotBuilder with the given main dataset
        alongside any region, grid, outline, point type datasets.

        :param main_dataset: The main dataset for this builder
        :type main_dataset: StrDict
        :param regions: A set of region-type datasets for this builder
        :type regions: Dict[str, StrDict]
        :param grids: A sed of grid-type datasets for this builder
        :type grids: Dict[str, StrDict]
        :param outlines: A set of outline-type datasets for this builder
        :type outlines: Dict[str, StrDict]
        :param points: A set of point-type datasets for this builder
        :type points: Dict[str, StrDict]
        """

        self._plot_status = PlotStatus.NO_DATA

        self._figure = Figure()
        if use_default_managers:
            self.update_figure(**self._default_figure_manager)
            # grids will all reference this manager
            self._grid_manager = deepcopy(self._default_grid_manager)
        else:
            self._grid_manager = {}

        self.use_default_managers = use_default_managers

        self._container = {
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        self._output_service = 'plotly'
        self.default_hex_resolution = 3

        if main_dataset is not None:
            self.set_main(**main_dataset)

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

    def set_main(
            self,
            data: DFType,
            latitude_field: str = None,
            longitude_field: str = None,
            hexbin_info: StrDict = None,
            manager: StrDict = None
    ):
        """Sets the main dataset to plot.

        :param data: The data for this set
        :type data: DFType
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param hexbin_info: A container for properties pertaining to hexagonal binning
        :type hexbin_info: StrDict
        :param manager: A container for the plotly properties for this dataset
        :type manager: StrDict
        """

        hbin_info = dict(hex_resolution=self.default_hex_resolution)
        if hexbin_info is None:
            hexbin_info = {}

        data = _read_data(data)
        dataset = dict(NAME='MAIN', RTYPE=data.RTYPE, DSTYPE='MN')
        data = _convert_latlong_data(data, latitude_field=latitude_field, longitude_field=longitude_field)

        hbin_info.update(hexbin_info)
        data = _convert_to_hexbin_data(data, **hbin_info)
        dataset['VTYPE'], dataset['data'], dataset['odata'] = data.VTYPE, data, data.copy(deep=True)
        dataset['manager'] = {}
        if self.use_default_managers:
            if dataset['VTYPE'] == 'NUM':
                _update_manager(dataset, deepcopy(self._default_quantitative_dataset_manager))
            else:
                _update_manager(dataset, deepcopy(self._default_qualitative_dataset_manager))
        _update_manager(dataset, **(manager if manager is not None else {}))
        self._container['main'] = dataset

    def _get_main(self) -> StrDict:
        """Retrieves the main dataset.

        Internal version.

        :return: The main dataset
        :rtype: StrDict
        """
        try:
            return self._container['main']
        except KeyError:
            raise gce.MainDatasetNotFoundError()

    def get_main(self):
        """Retrieves the main dataset.

        External version, returns a deepcopy.

        :return: The main dataset
        :rtype: StrDict
        """
        return deepcopy(self._get_main())

    def remove_main(self, pop: bool = False) -> StrDict:
        """Removes the main dataset.

        :param pop: Whether or not to return the removed dataset
        :type pop: bool
        :return: The removed dataset (pop=True)
        :rtype: StrDict
        """
        try:
            main = self._container.pop('main')
            if pop:
                return main
        except KeyError:
            raise gce.MainDatasetNotFoundError()

    def update_main_manager(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager the main dataset.

        :param updates: A dict containing updates for the dataset(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the dataset(s)
        :type kwargs: **kwargs
        """
        _update_manager(self._get_main(), updates=updates, overwrite=overwrite, **kwargs)

    def clear_main_manager(self):
        """Clears the manager of the main dataset.
        """
        try:
            self._get_main()['manager'] = {}
        except ValueError:
            raise gce.MainDatasetNotFoundError()

    def reset_main_data(self):
        """Resets the data within the main dataset to the data that was input at the beginning.
        """
        _reset_to_odata(self._get_main())

    """
    REGION FUNCTIONS
    """

    def add_region(
            self,
            name: str,
            data: DFType,
            manager: StrDict = None
    ):
        """Adds a region-type dataset to the builder.

        Region-type datasets should consist of Polygon-like geometries.
        Best results are read from a GeoDataFrame, or DataFrame.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param manager: The plotly properties for this dataset.
        :type manager: StrDict
        """
        _check_name(name)
        data = _read_data(data, allow_builtin=True)
        dataset = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='RGN', VTYPE=data.VTYPE)
        data = data[['value_field', 'geometry']]
        dataset['data'], dataset['odata'] = data, data.copy(deep=True)
        if self.use_default_managers:
            dataset['manager'] = deepcopy(self._default_region_manager)
        else:
            dataset['manager'] = {}
        _update_manager(dataset, **(manager or {}))
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
            raise gce.DatasetNotFoundError(name, "region")

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
            raise gce.DatasetNotFoundError(name, "region")

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

    def clear_region_manager(self, name: str = None):
        """Clears the manager of a region dataset.

        If the given name is none, clears all of the region managers.
        """
        self.update_region_manager(name=name, updates={}, overwrite=True)

    def reset_region_data(self, name: str = None):
        """Resets the data within the region dataset to the data that was input at the beginning.

        If the given name is None, all region datasets will be reset.

        :param name: The name of the dataset to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_regions().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_region(name))

    def reset_regions(self):
        """Resets the regions within the builder to empty.
        """
        self._container['regions'] = {}

    """
    GRID FUNCTIONS
    """

    def add_grid(
            self,
            name: str,
            data: DFType,
            hex_resolution: int = None,
            latitude_field: str = None,
            longitude_field: str = None
    ):
        """Adds a grid-type dataset to the builder.

        Grid-type datasets should consist of Polygon-like or Point-like geometries.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param hex_resolution: The hexagonal resolution to use for this dataset (None->builder default)
        :type hex_resolution: int
        :param latitude_field: The latitude column within the data
        :type latitude_field: str
        :param longitude_field: The longitude column within the data
        :type longitude_field: str
        """

        _check_name(name)
        data = _read_data(data, allow_builtin=True)
        dataset = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='GRD', VTYPE=data.VTYPE)
        data = _convert_to_hexbin_data(
            _convert_latlong_data(data, latitude_field=latitude_field, longitude_field=longitude_field),
            hex_resolution=hex_resolution or self.default_hex_resolution,
            binning_fn=lambda lst: 0
        )
        data = data[['value_field', 'geometry']]
        dataset['data'], dataset['odata'] = data, data.copy(deep=True)
        dataset['manager'] = self._grid_manager
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
            raise gce.DatasetNotFoundError(name, "grid")

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
            raise gce.DatasetNotFoundError(name, "grid")

    def update_grid_manager(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the general grid manager.

        :param updates: A dict of updates for the manager
        :type updates: StrDict
        :param overwrite: Whether or not to override existing manager properties
        :type overwrite: bool
        :param kwargs: Any additional updates for the manager
        :type kwargs: **kwargs
        """

        if self._get_grids():
            dct = {'manager': deepcopy(self._grid_manager)}
            _update_manager(dct, updates=updates, overwrite=overwrite, **kwargs)
            self._grid_manager.clear()
            self._grid_manager.update(dct['manager'])

    def clear_grid_manager(self):
        """Clears the manager of a region dataset.

        If the given name is none, clears all of the region managers.
        """
        if self._get_grids():
            self._grid_manager.clear()

    def reset_grid_data(self, name: str = None):
        """Resets the data within the grid dataset to the data that was input at the beginning.

        If the given name is None, all grid datasets will be reset.

        :param name: The name of the dataset to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_grids().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_grid(name))

    def reset_grids(self):
        """Resets the grid dataset container to it's original state.
        """
        self._container['grids'] = {}
        self.clear_grid_manager()

    """
    OUTLINE FUNCTIONS
    """

    def add_outline(
            self,
            name: str,
            data: DFType,
            latitude_field: str = None,
            longitude_field: str = None,
            as_boundary: bool = False,
            manager: StrDict = None
    ):
        """Adds a outline-type dataset to the builder.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param as_boundary: Changes the data into one big boundary if true
        :type as_boundary: bool
        :param manager: Plotly properties for this dataset
        :type manager: StrDict
        """
        _check_name(name)
        data = _read_data(data, allow_builtin=True)
        dataset = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='OUT', VTYPE=data.VTYPE)
        data = _convert_latlong_data(data, latitude_field=latitude_field,
                                     longitude_field=longitude_field)[['value_field', 'geometry']]
        if as_boundary:
            data = gcg.unify_geodataframe(data)

        dataset['data'], dataset['odata'] = data, data.copy(deep=True)
        if self.use_default_managers:
            dataset['manager'] = deepcopy(self._default_outline_manager)
        else:
            dataset['manager'] = {}
        _update_manager(dataset, **(manager or {}))
        self._get_outlines()[name] = dataset

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
            raise gce.DatasetNotFoundError(name, "outline")

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
            raise gce.DatasetNotFoundError(name, "outline")

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

    def clear_outline_manager(self, name: str = None):
        """Clears the manager of a outline dataset.

        If the given name is none, clears all of the outline managers.
        """
        self.update_outline_manager(name=name, updates={}, overwrite=True)

    def reset_outline_data(self, name: str = None):
        """Resets the data within the outline dataset to the data that was input at the beginning.

        If the given name is None, all outline datasets will be reset.

        :param name: The name of the dataset to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_outlines().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_outline(name))

    def reset_outlines(self):
        """Resets the outlines within the builder to empty.
        """
        self._container['outlines'] = {}

    """
    POINT FUNCTIONS
    """

    def add_point(
            self,
            name: str,
            data: DFType,
            latitude_field: str = None,
            longitude_field: str = None,
            manager: StrDict = None
    ):
        """Adds a outline-type dataset to the builder.

        Ideally the dataset's 'data' member should contain
        lat/long columns or point like geometry column. If the geometry column
        is present and contains no point like geometry, the geometry will be converted
        into a bunch of points.

        :param name: The name this dataset is to be stored with
        :type name: str
        :param data: The location of the data for this dataset
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param manager: Plotly properties for this dataset
        :type manager: StrDict
        """

        _check_name(name)
        data = _read_data(data, allow_builtin=False)
        dataset = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='PNT', VTYPE=data.VTYPE)
        data = _convert_latlong_data(data, latitude_field=latitude_field,
                                     longitude_field=longitude_field)[['value_field', 'geometry']]

        dataset['data'], dataset['odata'] = data, data.copy(deep=True)
        if self.use_default_managers:
            dataset['manager'] = deepcopy(self._default_point_manager)
        else:
            dataset['manager'] = {}
        _update_manager(dataset, **(manager or {}))
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
            return self._get_points()[name]
        except KeyError:
            raise gce.DatasetNotFoundError(name, "point")

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
            raise gce.DatasetNotFoundError(name, "point")

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

    def clear_point_manager(self, name: str = None):
        """Clears the manager of a point dataset.

        If the given name is none, clears all of the point managers.
        """
        self.update_point_manager(name=name, updates={}, overwrite=True)

    def reset_point_data(self, name: str = None):
        """Resets the data within the point dataset to the data that was input at the beginning.

        If the given name is None, all point datasets will be reset.

        :param name: The name of the dataset to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_points().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_point(name))

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
            raise gce.MainDatasetNotFoundError()

        if add_to_plot:
            empties = dataset['data'][dataset['data']['value_field'] == empty_symbol]
            if not empties.empty:
                empties['value_field'] = 0
                self.get_grids()['|*EMPTY*|'] = empties
        dataset['data'] = dataset['data'][dataset['data']['value_field'] != empty_symbol]

    # this is both a data altering, and plot altering function
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
            dataset = self._get_main()
        except ValueError:
            raise gce.MainDatasetNotFoundError()

        if dataset['VTYPE'] == 'STR':
            raise TypeError("A qualitative dataset can not be converted into a logarithmic scale.")
        _update_manager(dataset, butil.logify_scale(dataset['data'], **kwargs))

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

        datas = self.apply_to_query(to, lambda dataset: dataset['data'])

        def gpdhelp(dataset: dict):
            dataset['data'] = butil.gpd_clip(dataset['data'], datas, validate=True)

        def sjoinhelp(dataset: dict):
            dataset['data'] = butil.sjoin_clip(dataset['data'], datas, operation=operation, validate=True)

        if method == 'gpd':
            self.apply_to_query(clip, gpdhelp)
        elif method == 'sjoin':
            self.apply_to_query(clip, sjoinhelp)
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

    def adjust_colorbar_size(self):
        """Adjusts the color scale position of the color bar to match the plot area size.
        """
        dataset = self._get_main()
        layout = self._figure.layout
        layout.yaxis.automargin = False
        margin_b, margin_t = layout.margin.b, layout.margin.t
        fig_h, fig_w = layout.height, layout.width

        barlen = fig_h-margin_t + 16
        print(barlen)
        print(margin_b, margin_t)
        print(fig_h, fig_w)

        up = dict(colorbar=dict(lenmode='pixels', yanchor='bottom', y=0, ypad=0))

        _update_manager(dataset, updates=up)
        # (2.1228468243122216, 50.73231097463263)
        #
        # (41.675105088867326, 83.23324000000001)
        lrange = layout.geo.lataxis.range
        print(lrange)


        print(dataset['manager'])
        self._figure.add_annotation(text='BL', xref='paper', yref='paper', x=0, y=0)
        self._figure.add_annotation(text='TL', xref='paper', yref='paper', x=0, y=1)
        self._figure.add_annotation(text='BR', xref='paper', yref='paper', x=1, y=0)
        self._figure.add_annotation(text='TR', xref='paper', yref='paper', x=1, y=1)

        self._figure.add_annotation(text='MT', xref='paper', yref='paper', x=0.8, y=(fig_h-margin_t)/fig_h)
        self._figure.add_annotation(text='MB', xref='paper', yref='paper', x=0.8, y=margin_b/fig_h)

    # TODO: This function should only apply to the main dataset (maybe, think more)
    def adjust_opacity(self, alpha: float = None):
        """Conforms the opacity of the color bar of the main dataset to an alpha value.

        The alpha value can be passed in as a parameter, otherwise it is taken
        from the marker.opacity property within the dataset's manager.

        :param alpha: The alpha value to conform the color scale to
        :type alpha: float
        """

        dataset = self._get_main()
        butil.opacify_colorscale(dataset, alpha=alpha)

    def adjust_focus(self, on: str = 'main', center_on: bool = False, rotation_on: bool = True, ranges_on: bool = True,
                     buffer_lat: tuple = (0, 0), buffer_lon: tuple = (0, 0), validate: bool = False):
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

    def auto_grid(self, on: str = 'main', by_bounds: bool = False, hex_resolution: int = None):
        """Makes a grid over the queried datasets.

        :param on: The query for the datasets to have a grid generated over them
        :type on: str
        :param by_bounds: Whether or not to treat the  geometries as a single boundary
        :type by_bounds: bool
        :param hex_resolution: The hexagonal resolution to use for the auto grid
        :type hex_resolution: int
        """

        fn = gcg.generate_grid_over if by_bounds else gcg.hexify_dataframe
        hex_resolution = hex_resolution if hex_resolution is not None else self.default_hex_resolution

        def helper(dataset):
            return fn(dataset['data'], hex_resolution=hex_resolution)

        try:
            grid = GeoDataFrame(pd.concat(list(self.apply_to_query(on, helper))), crs='EPSG:4326')[
                ['value_field', 'geometry']].drop_duplicates()
            if not grid.empty:
                grid['value_field'] = 0
                self._get_grids()[f'|*AUTO-{on}*|'] = {'data': grid, 'manager': self._grid_manager}
            else:
                raise ValueError("There may have been an error when generating auto grid, shapes may span too large "
                                 "of an area.")

        except ValueError:
            pass

    def discretize_scale(self, scale_type: str = 'sequential', **kwargs):
        """Converts the color scale of the dataset(s) to a discrete scale.

        :param scale_type: One of 'sequential', 'discrete' for the type of color scale being used
        :type scale_type: str
        :param kwargs: Keyword arguments to be passed into the discretize functions
        :type kwargs: **kwargs
        """

        try:
            dataset = self._get_main()
        except ValueError:
            raise gce.MainDatasetNotFoundError()

        if dataset['VTYPE'] == 'STR':
            raise ValueError(f"You can not discretize a qualitative dataset.")
        else:
            low = dataset['manager'].get('zmin', min(dataset['data']['value_field']))
            high = dataset['manager'].get('zmax', max(dataset['data']['value_field']))
            dataset['manager']['colorscale'] = discretize_cscale(dataset['manager'].get('colorscale'), scale_type,
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
            return self._get_main()
        elif query in ['regions', 'grids', 'outlines', 'points', 'all']:
            if big_query:
                return self._container if query == 'all' else self._container[query]
            else:
                raise gce.BuilderQueryInvalidError("The given query should not refer to a collection of datasets.")

        try:
            typer, name = _split_name(query)
        except ValueError:
            raise gce.BuilderQueryInvalidError(
                "The given query should be one of ['regions', 'grids', 'outlines', 'points', "
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
            raise gce.BuilderQueryInvalidError("The given dataset type does not exist. Must be one of ['region', "
                                               f"'grid', 'outline', 'point']. Received {typer}.")

    get_query_data = lambda self, name: self.apply_to_query(name, lambda ds: ds['data'])

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
            raise gce.NoDatasetsError("There are no region-type datasets to plot.")
        for regname, regds in self._get_regions().items():
            self._figure.add_trace(_prepare_choropleth_trace(
                gcg.conform_geogeometry(regds['data']),
                mapbox=mapbox).update(regds['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_grids(self, remove_underlying: bool = False):
        """Plots the grid datasets within the builder.

        Merges all of the datasets together, and plots it as a single plot trace.
        """
        if not self._get_grids():
            raise gce.NoDatasetsError("There are no grid-type datasets to plot.")

        merged = gcg.conform_geogeometry(
            pd.concat(self.apply_to_query('grids', lambda dataset: dataset['data'])).drop_duplicates())

        if remove_underlying:
            merged = self._remove_underlying_grid(self._get_main()['data'], merged)
        self._figure.add_trace(_prepare_choropleth_trace(
            merged,
            mapbox=self.plot_output_service == 'mapbox').update(text='GRID').update(self._grid_manager))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_main(self):
        """Plots the main dataset within the builder.

        If qualitative, the dataset is split into uniquely labelled plot traces.
        """
        try:
            dataset = self._get_main()
        except gce.MainDatasetNotFoundError:
            raise gce.NoDatasetsError("There is no main dataset to plot.")
        df = gcg.conform_geogeometry(dataset['data'])

        # qualitative dataset
        if dataset['VTYPE'] == 'STR':
            df['text'] = 'BEST OPTION: ' + df['value_field']
            colorscale = dataset['manager'].pop('colorscale')
            try:
                colorscale = get_scale(colorscale)
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
                        raise gce.ColorscaleError(
                            "If the colorscale is a map, you must provide hues for each option.")
                    self._figure.add_trace(_prepare_choropleth_trace(
                        v,
                        mapbox=mapbox).update(name=k, showscale=False, showlegend=True, text=v['text']).update(manager))

            elif isinstance(colorscale, list) or isinstance(colorscale, tuple):

                for i, (k, v) in enumerate(sep.items()):

                    try:
                        if isinstance(colorscale[i], list) or isinstance(colorscale[i], tuple):
                            manager['colorscale'] = solid_scale(colorscale[i][1])
                        else:
                            manager['colorscale'] = solid_scale(colorscale[i])
                    except IndexError:
                        raise IndexError("There were not enough hues for all of the unique options in the dataset.")
                    self._figure.add_trace(
                        _prepare_choropleth_trace(v, mapbox=mapbox).update(name=k, showscale=False, showlegend=True,
                                                                           text=v['text']).update(manager))
            else:
                raise gce.ColorscaleError("If the colorscale is a map, you must provide hues for each option.")
        # quantitative dataset
        else:
            df['text'] = 'VALUE: ' + df['value_field'].astype(str)
            self._figure.add_trace(_prepare_choropleth_trace(df, mapbox=self.plot_output_service == 'mapbox').update(
                text=df['text']).update(dataset['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_outlines(self, raise_errors: bool = False):
        """Plots the outline datasets within the builder.

        All of the outlines are treated as separate plot traces.
        The datasets must first be converted into point-like geometries.

        :param raise_errors: Whether or not to throw errors upon reaching empty dataframes
        :type raise_errors: bool
        """
        if not self._get_outlines():
            raise gce.NoDatasetsError("There are no outline-type datasets to plot.")

        mapbox = self.plot_output_service == 'mapbox'
        for outname, outds in self._get_outlines().items():
            self._figure.add_trace(_prepare_scattergeo_trace(
                gcg.pointify_geodataframe(outds['data'].explode(),
                                          keep_geoms=False,
                                          raise_errors=raise_errors),
                separate=True,
                disjoint=True,
                mapbox=mapbox).update(outds['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_points(self):
        """Plots the point datasets within the builder.

        All of the point are treated as separate plot traces.
        """

        if not self._get_points():
            raise gce.NoDatasetsError("There are no point-type datasets to plot.")

        mapbox = self.plot_output_service == 'mapbox'
        for poiname, poids in self._get_points().items():
            self._figure.add_trace(_prepare_scattergeo_trace(
                poids['data'],
                separate=False,
                disjoint=False,
                mapbox=mapbox).update(poids['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def set_mapbox(self, accesstoken: str):
        """Prepares the builder for a mapbox output.

        Sets figure.layout.mapbox_accesstoken, and plot_settings output service.

        :param accesstoken: A mapbox access token for the plot
        :type accesstoken: str
        """
        self.plot_output_service = 'mapbox'
        self._figure.update_layout(mapbox_accesstoken=accesstoken)

    def build_plot(
            self,
            plot_regions: bool = True,
            plot_grids: bool = True,
            plot_main: bool = True,
            plot_outlines: bool = True,
            plot_points: bool = True,
            raise_errors: bool = False
    ):
        """Builds the final plot by adding traces in order.

        Invokes the functions in the following order:
        1) plot regions
        2) plot grids
        3) plot dataset
        4) plot outlines
        5) plot points

        In the future we should alter these functions to
        allow trace order implementation.

        :param plot_regions: Whether or not to plot region datasets
        :type plot_regions: bool
        :param plot_grids: Whether or not to plot grid datasets
        :type plot_grids: bool
        :param plot_main: Whether or not to plot the main dataset
        :type plot_main: bool
        :param plot_outlines: Whether or not to plot outline datasets
        :type plot_outlines: bool
        :param plot_points: Whether or not to plot point datasets
        :type plot_points: bool
        :param raise_errors: Whether or not to raise errors related to empty dataset collections
        :type raise_errors: bool
        """
        if plot_regions:
            try:
                self.plot_regions()
            except gce.NoDatasetsError as e:
                if raise_errors:
                    raise e
        if plot_grids:
            try:
                self.plot_grids(remove_underlying=True)
            except gce.NoDatasetsError as e:
                if raise_errors:
                    raise e
        if plot_main:
            try:
                self.plot_main()
            except gce.NoDatasetsError as e:
                if raise_errors:
                    raise e
        if plot_outlines:
            try:
                self.plot_outlines()
            except gce.NoDatasetsError as e:
                if raise_errors:
                    raise e
        if plot_points:
            try:
                self.plot_points()
            except gce.NoDatasetsError as e:
                if raise_errors:
                    raise e

    def output_figure(self, filepath: str, clear_figure: bool = False, **kwargs):
        """Outputs the figure to a filepath.

        The figure is output via Plotly's write_image() function.
        Plotly's Kaleido is required for this feature.

        :param filepath: The filepath to output the figure at (including filename and extension)
        :type filepath: str
        :param clear_figure: Whether or not to clear the figure after this operation
        :type clear_figure: bool
        :param kwargs: Keyword arguments for the write_image function
        :type kwargs: **kwargs
        """
        self._figure.write_image(filepath, **kwargs)
        if clear_figure:
            self.clear_figure()

    def display_figure(self, clear_figure: bool = False, **kwargs):
        """Displays the figure.

        The figure is displayed via Plotly's show() function.
        Extensions may be needed.

        :param clear_figure: Whether or not to clear the figure after this operation
        :type clear_figure: bool
        :param kwargs: Keyword arguments for the show function
        :type kwargs: **kwargs
        """
        self._figure.show(**kwargs)
        if clear_figure:
            self.clear_figure()

    def get_plot_status(self) -> PlotStatus:
        """Retrieves the status of the internal plot.

        :return: The status of the plot
        :rtype: PlotStatus
        """
        return self._plot_status

    def clear_figure(self):
        """Clears the figure of its current data.
        """
        self._plot_status = PlotStatus.NO_DATA
        self._figure.data = []

    def reset(self):
        """Resets the builder to it's initial state.
        """

        self._plot_status = PlotStatus.NO_DATA
        self._figure = Figure()

        if self.use_default_managers:
            self.update_figure(**self._default_figure_manager)
            # grids will all reference this manager
            self._grid_manager = deepcopy(self._default_grid_manager)
        else:
            self._grid_manager = {}

        self._container = {
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        self._output_service = 'plotly'
        self.default_hex_resolution = 3

    def reset_data(self):
        """Resets the datasets of the builder to their original state.
        """
        try:
            self.reset_main_data()
        except gce.MainDatasetNotFoundError:
            pass
        self.reset_region_data()
        self.reset_grid_data()
        self.reset_outline_data()
        self.reset_point_data()
