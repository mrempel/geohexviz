from copy import deepcopy
from enum import Enum
import os
from typing import Any, Tuple, Dict, Union, Callable

import fiona.errors
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.io as pio
from geopandas import GeoDataFrame
from pandas import DataFrame
from plotly.graph_objs import Figure, Choropleth, Scattergeo, Choroplethmapbox, Scattermapbox

from geohexviz.templates import get_template
from geohexviz.utils import geoutils as gcg
from geohexviz.utils import plot_util as butil
from geohexviz.utils.colorscales import solid_scale, discretize_cscale, \
    get_scale
from geohexviz.utils.util import fix_filepath, get_column_type, \
    simplify_dicts, dict_deep_update, get_percdiff, parse_args_kwargs, get_best, get_worst
from geohexviz.errors import *
from functools import reduce
import warnings

_read_method_mapping = {
    'geopandas': gpd.read_file,
    'csv': pd.read_csv,
    'excel': pd.read_excel
}

_extension_mapping = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    '.shp': gpd.read_file,
    '.kml': gpd.read_file,
    '.gpkg': gpd.read_file,
    '.geojson': gpd.read_file,
    '': gpd.read_file
}

_group_functions_short = {
    'count': lambda lst: len(lst),
    'sum': lambda lst: sum(lst),
    'avg': lambda lst: sum(lst) / len(lst),
    'min': lambda lst: min(lst),
    'max': lambda lst: max(lst),
    'best': get_best,
    'worst': get_worst
}

_group_functions = {
    'count': _group_functions_short['count'],
    'occ': _group_functions_short['count'],
    'occurrences': _group_functions_short['count'],
    'sum': _group_functions_short['sum'],
    'summation': _group_functions_short['sum'],
    'avg': _group_functions_short['avg'],
    'average': _group_functions_short['avg'],
    'min': _group_functions_short['min'],
    'minimum': _group_functions_short['min'],
    'max': _group_functions_short['max'],
    'maximum': _group_functions_short['max'],
    'best': _group_functions_short['best'],
    'best_option': _group_functions_short['best'],
    'mfreq': _group_functions_short['best'],
    'most_frequent': _group_functions_short['best'],
    'worst': _group_functions_short['worst'],
    'worst_option': _group_functions_short['worst'],
    'lfreq': _group_functions_short['worst'],
    'least_frequent': _group_functions_short['worst']
}

StrDict = Dict[str, Any]
DFType = Union[str, DataFrame, GeoDataFrame]

# add driver for KML files
gpd.io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
gpd.io.file.fiona.drvsupport.supported_drivers['libkml'] = 'rw'
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
gpd.io.file.fiona.drvsupport.supported_drivers['kml'] = 'rw'

def _reset_to_odata(layer: StrDict):
    """Resets the odata parameter of a layer.

    :param layer: The layer to reset
    :type layer: StrDict
    """
    layer['data'] = layer['odata'].copy(deep=True)


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


def _validate_layer(layer: StrDict):
    """Validates a layer.

    :param layer: The layer to validate
    :type layer: StrDict
    """
    if 'data' not in layer:
        raise ValueError("There must be a 'data' member present in the layer.")


def _get_reader_function_from_method(method: str):
    try:
        return _read_method_mapping[method]
    except KeyError:
        raise ValueError("The input read method was not valid."
                         f"\n Available file read functions: {list(_read_method_mapping.keys())}")


def _get_reader_function_from_path(ext: str):
    try:
        return _extension_mapping[ext]
    except KeyError:
        raise ValueError("The input filepath had an incorrect extension,"
                         " this project only supports some types of files.")


def _read_data_file_dict(data):
    if isinstance(data, dict):
        try:
            dpath = data.pop("path")
        except KeyError:
            raise DataTypeError("There must be a 'path' parameter"
                                " present when passing the 'data' parameter as a dict.")
        read_method = data.pop("method", None)
        read_args, read_kwargs = parse_args_kwargs(data)
        return _read_data_file(dpath, read_method=read_method,
                               read_args=read_args, **read_kwargs)
    return _read_data_file(data)


def _read_data_file(data: str, read_method=None, read_args=None, **kwargs) -> DataFrame:
    """Reads data from a file, based on extension.

    :DEPRECATED:

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
    if read_args is None:
        read_args = ()
    try:
        filepath, extension = os.path.splitext(os.path.join(os.path.dirname(__file__), data))
        filepath = f"{filepath}{extension}"

        try:
            read_fn = _get_reader_function_from_method(read_method)
        except ValueError as e:
            if read_method is not None:
                raise e
            try:
                read_fn = _get_reader_function_from_path(extension)
            except ValueError:
                read_fn = gpd.read_file

        try:
            data = read_fn(data, *read_args, **kwargs)
        except TypeError:
            raise DataFileReadError("The read function is not a callable object.")

        except fiona.errors.DriverError:
            raise DataFileReadError("The file could not be read, check the filepath.")
        try:
            if not data.crs:
                data.crs = 'EPSG:4326'
            else:
                data.to_crs(crs='EPSG:4326', inplace=True)
        except AttributeError:
            pass

    except TypeError:
        pass
    return data

valid_data_intypes = ['region', 'dataframe', 'geodataframe', 'file']

def _read_data_full(
        name: str,
        dstype: LayerType,
        data: DFType,
        allow_builtin: bool = False
):
    """Reads data from a file, based on extension.

    If the file extension is unknown the file is passed
    directly into geopandas.read_file().

    This function uses both geopandas and pandas to read data.

    In the future it may be beneficial to allow the reading
    of databases, and feather.

    :param name: The name of the layer containing the given data
    :type name: str
    :param dstype: The type of layer containing the given data
    :type dstype: LayerType
    :param data: The data to be read
    :type data: DFType
    :param allow_builtin: Whether to allow builtin data types or not (countries, continents)
    :type allow_builtin: bool
    :return: The read data
    :rtype: DataFrame
    """

    read_args = ()
    read_kwargs = {}
    rtype = 'file'

    if isinstance(data, dict):
        odata = deepcopy(data)
        if "path" in odata:
            data = odata.pop("path")
            read_args, read_kwargs = parse_args_kwargs(odata)

    # check if string
    if isinstance(data, str):
        filepath, extension = os.path.splitext(os.path.join(os.path.dirname(__file__), data))

        try:
            read_fn = _get_reader_function_from_path(extension)
        except ValueError:
            read_fn = gpd.read_file

        try:
            data = read_fn(data, *read_args, **read_kwargs)
        except Exception as e:
            # TODO: to know if this was a region name or file path, use the os library.
            if allow_builtin:
                try:
                    data, rtype = butil.get_shapes_from_world(data), 'builtin'
                except (KeyError, ValueError, TypeError):
                    raise DataReadError(
                        name, dstype,
                        message="If the 'data' property passed was a filepath it was not read.\n"
                                f"File Read Error:\n{str(e)}\n"
                                "If the 'data' property passed was a region name "
                                "(country or continent CAPS), it was invalid.\n"
                                "Ensure that the region name is spelled out in full and is in all CAPS."
                    )
            else:
                raise DataFileReadError(
                    name, dstype,
                    message="The file could not be read.\n"
                            f"Error: {str(e)}"
                )


    try:
        data = GeoDataFrame(data)
        data['value_field'] = 0
        data.RTYPE = rtype
        data.VTYPE = 'NUM'
    except Exception:
        raise DataTypeError(name, dstype, allow_builtin=allow_builtin)

    return data





def _read_data(name: str, dstype: LayerType, data: DFType, allow_builtin: bool = False) -> GeoDataFrame:
    """Reads the data into a usable type for the builder.

    :param data: The data to be read
    :type data: DFType
    :param allow_builtin: Whether to allow builtin data types or not (countries, continents)
    :type allow_builtin: bool
    :return: A proper geodataframe from the input data
    :rtype: GeoDataFrame
    """
    rtype = 'frame'
    try:
        data, rtype = _read_data_file_dict(data), 'file'
    except DataFileReadError as e:
        if allow_builtin:
            try:
                data, rtype = butil.get_shapes_from_world(data), 'builtin'
            except (KeyError, ValueError, TypeError):
                pass
        elif not isinstance(data, DataFrame):
            raise e

    if isinstance(data, DataFrame):
        data = GeoDataFrame(data)
        data['value_field'] = 0
        data.RTYPE = rtype
        data.VTYPE = 'NUM'
    else:
        raise DataTypeError(name, dstype, allow_builtin)
    return data


def _parse_latlong_fields(name: str, dstype: LayerType, data: GeoDataFrame,
                          latitude_field: str = None, longitude_field: str = None) -> Tuple[pd.Series, pd.Series]:
    """Parses the lat/long columns from a GeoDataFrame (called upon loading).

    :param data: The data to parse columns from
    :type data: GeoDataFrame
    :param latitude_field: The name of the latitude column (or none)
    :type latitude_field: str
    :param longitude_field: The name of the longitude column (or none)
    :type longitude_field: str
    :return: The two columns if applicable (or none, none)
    :rtype: Tuple[pd.Series, pd.Series]
    """

    if "geometry" in data.columns and latitude_field is None and longitude_field is None:
        return None, None

    try:
        latitude_field = data[latitude_field]
    except KeyError:
        for col in data.columns:
            if col.lower() in latitude_aliases:
                latitude_field = data[col]
                break
        if latitude_field is None:
            raise GeometryParseLatLongError(name, dstype, True)

    try:
        longitude_field = data[longitude_field]
    except KeyError:
        for col in data.columns:
            if col.lower() in longitude_aliases:
                longitude_field = data[col]
                break
        if longitude_field is None:
            raise GeometryParseLatLongError(name, dstype, False)

    if get_column_type(data, latitude_field.name) != "NUM":
        try:
            latitude_field = latitude_field.astype(float)
        except (ValueError, TypeError):
            raise LatLongParseTypeError(name, dstype, True)

    if get_column_type(data, longitude_field.name) != "NUM":
        try:
            longitude_field = longitude_field.astype(float)
        except (ValueError, TypeError):
            raise LatLongParseTypeError(name, dstype, False)

    return latitude_field, longitude_field


def _convert_latlong_data(name: str, dstype: LayerType, data: GeoDataFrame,
                          latitude_field: str = None, longitude_field: str = None) -> GeoDataFrame:
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
        raise DataEmptyError("If the data passed is a DataFrame/GeoDataFrame, it must not be empty.")

    data = data.copy(deep=True)

    latitude_field, longitude_field = _parse_latlong_fields(
        name, dstype,
        data, latitude_field=latitude_field, longitude_field=longitude_field)

    try:
        data = GeoDataFrame(data, geometry=gpd.points_from_xy(longitude_field, latitude_field, crs='EPSG:4326'))
    except (TypeError, ValueError):
        pass

    # TODO: test this with empty dataframes
    # removes empty latitude and longitude entries that cause errors in the final product
    if latitude_field is not None and longitude_field is not None:
        data.dropna(subset=[latitude_field.name, longitude_field.name], inplace=True)

    gcg.conform_geogeometry(data)
    data.vtype = 'NUM'
    return data


def _convert_to_hexbin_data(name: str, dstype: LayerType, data: GeoDataFrame, hex_resolution: int, binning_args=None,
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

    if isinstance(binning_fn, dict):
        fn = binning_fn.pop('fn', binning_fn)
        binning_args, kwargs = parse_args_kwargs(binning_fn, default_args=binning_args, default_kwargs=kwargs)
        binning_fn = fn

    try:
        binning_fn = _group_functions[str(binning_fn)]
    except KeyError:
        pass
    vtype = 'NUM' if binning_field is None else get_column_type(data, binning_field)
    if vtype == 'UNK':
        raise BinValueTypeError("An invalid binning column was passed.\n"
                                "The binning field must be string or numerical column.")
    elif vtype == 'STR':
        data[binning_field] = data[binning_field].astype(str)

    if binning_fn is None:
        binning_fn = _group_functions['best'] if vtype == 'STR' else _group_functions['sum']

    data = gcg.bin_by_hexid(data, binning_field=binning_field, binning_fn=binning_fn, binning_args=binning_args,
                            result_name='value_field', add_geoms=True, **kwargs)
    vtype = get_column_type(data, 'value_field')

    # if the resulting column is of unknown type, the binning function did not produce a viable result
    if vtype == 'UNK':
        raise BinValueTypeError("The result of the binning operation was invalid.\n"
                                "The binning operation must produce a column that contains either string, or numbers.")
    gcg.conform_geogeometry(data)
    data.VTYPE = vtype

    # if there is no data after the above process, there was no grid generated
    if data.empty:
        raise NoHexagonalTilingError(name, dstype)
    return data


def _update_manager(layer: StrDict, updates: StrDict = None, overwrite: bool = False, **kwargs):
    """Updates the manager of a given layer.

    :param layer: The layer whose manager to update
    :type layer: StrDict
    :param updates: A dict of updates for the manager
    :type updates: StrDict
    :param overwrite: Whether to overwrite the existing manager with the new one or not
    :type overwrite: bool
    :param kwargs: Extra updates for the manager
    :type kwargs: **kwargs
    """
    updates = simplify_dicts(fields=updates, **kwargs)
    if overwrite:
        layer['manager'] = updates
    else:
        dict_deep_update(layer['manager'], updates)


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


def _split_query(query: str) -> Tuple[str, str]:
    """Splits a query into the layer type and layer name.

    :param query: The query to split
    :type query: str
    :return: The split query (type, name)
    :rtype: Tuple[str, str]
    """
    lind = query.index(':')
    return query[:lind], query[lind + 1:]


def _check_name(name: str, dstype: LayerType):
    """Checks if the given layer name is valid, and throws an error if it is not.

    :param name: The name of the layer
    :type name: str
    """
    if any(not i.isalnum() and i not in ('_', ' ') for i in name):
        raise LayerNamingError(name, dstype, "Non-alphanumeric found (besides underscores)")


class PlotStatus(Enum):
    """An enumeration of different plot status.
    """
    DATA_PRESENT = 0
    NO_DATA = 1


class PlotBuilder:
    """This class contains a Builder implementation for visualizing Plotly Hex data.
    """

    def __init__(self, hexbin_layer: StrDict = None, regions: Dict[str, StrDict] = None,
                 grids: Dict[str, StrDict] = None, outlines: Dict[str, StrDict] = None,
                 points: Dict[str, StrDict] = None, use_templates: bool = True):
        """Initializer for instances of PlotBuilder.

        Initializes a new PlotBuilder with the given main layer
        alongside any region, grid, outline, point type layers.

        :param hexbin_layer: The main layer for this builder
        :type hexbin_layer: StrDict
        :param regions: A set of region-type layers for this builder
        :type regions: Dict[str, StrDict]
        :param grids: A sed of grid-type layers for this builder
        :type grids: Dict[str, StrDict]
        :param outlines: A set of outline-type layers for this builder
        :type outlines: Dict[str, StrDict]
        :param points: A set of point-type layers for this builder
        :type points: Dict[str, StrDict]
        """

        self._plot_status = PlotStatus.NO_DATA

        self._figure = Figure()
        if use_templates:
            self._figure.update(get_template('figure'))
            # grids will all reference this manager
            self._grid_manager = deepcopy(get_template('grid'))
        else:
            self._grid_manager = {}

        self.use_templates = use_templates

        self._container = {
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        self._output_service = 'plotly'
        self.default_hex_resolution = 3
        self.output_destination = None
        self._last_output_location = None  # for future use

        if hexbin_layer is not None:
            self.set_hexbin(**hexbin_layer)

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

    def __getitem__(self, item):
        """getitem method works like search method.

        :param item: The item to search for
        :type item: str
        :return: The retrieved layer(s)
        :rtype: StrDict
        """
        return self.search(item)

    """
    MAIN layer FUNCTIONS
    """

    def set_hexbin(
            self,
            data: DFType,
            latitude_field: str = None,
            longitude_field: str = None,
            hex_resolution: int = None,
            hexbin_info: StrDict = None,
            manager: StrDict = None
    ):
        """Sets the hexbin layer to plot.

        :param data: The data for this set
        :type data: DFType
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param hex_resolution: The hex resolution to use (this can also be passed via hexbin_info)
        :type hex_resolution: int
        :param hexbin_info: A container for properties pertaining to hexagonal binning
        :type hexbin_info: StrDict
        :param manager: A container for the plotly properties for this layer
        :type manager: StrDict
        """
        self._get_grids().pop('|*EMPTY*|', None)

        if hexbin_info is None:
            hexbin_info = {}

        selected_res = (hex_resolution or hexbin_info.get('hex_resolution')) or self.default_hex_resolution
        hbin_info = dict(hex_resolution=selected_res)

        data = _read_data_full('hexbin', LayerType.HEXBIN, data)
        layer = dict(NAME='HEXBIN', RTYPE=data.RTYPE, DSTYPE='HEX', HRES=selected_res)
        data = _convert_latlong_data('hexbin', LayerType.HEXBIN, data,
                                     latitude_field=latitude_field, longitude_field=longitude_field)

        hbin_info.update(hexbin_info)
        data = _convert_to_hexbin_data("hexbin", LayerType.HEXBIN, data, **hbin_info)
        layer['VTYPE'], layer['data'], layer['odata'] = data.VTYPE, data, data.copy(deep=True)
        layer['manager'] = {}
        if self.use_templates:
            if layer['VTYPE'] == 'NUM':
                _update_manager(layer, deepcopy(get_template('main_quant')))
            else:
                _update_manager(layer, deepcopy(get_template('main_qual')))
        _update_manager(layer, **(manager if manager is not None else {}))
        self._container['hexbin'] = layer

    def _get_hexbin(self) -> StrDict:
        """Retrieves the main layer.

        Internal version.

        :return: The main layer
        :rtype: StrDict
        """
        try:
            return self._container['hexbin']
        except KeyError:
            raise NoLayerError("hexbin", LayerType.HEXBIN)

    def get_hexbin(self):
        """Retrieves the main layer.

        External version, returns a deepcopy.

        :return: The main layer
        :rtype: StrDict
        """
        return deepcopy(self._get_hexbin())

    def remove_hexbin(self, pop: bool = False) -> StrDict:
        """Removes the main layer.

        :param pop: Whether or not to return the removed layer
        :type pop: bool
        :return: The removed layer (pop=True)
        :rtype: StrDict
        """
        try:
            hexbin = self._container.pop('hexbin')

            # remove the empty grid that may or may not have been added
            try:
                self.remove_grid('|*EMPTY*|')
            except NoLayerError:
                pass

            if pop:
                return hexbin
        except KeyError:
            raise NoLayerError("hexbin", LayerType.HEXBIN)

    def update_hexbin_manager(self, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager the hexbin layer.

        :param updates: A dict containing updates for the layer(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the layer(s)
        :type kwargs: **kwargs
        """
        _update_manager(self._get_hexbin(), updates=updates, overwrite=overwrite, **kwargs)

    def clear_hexbin_manager(self):
        """Clears the manager of the hexbin layer.
        """
        try:
            self._get_hexbin()['manager'] = {}
        except ValueError:
            raise NoLayerError("hexbin", LayerType.HEXBIN)

    def reset_hexbin_data(self):
        """Resets the data within the hexbin layer to the data that was input at the beginning.
        """
        _reset_to_odata(self._get_hexbin())

    """
    REGION FUNCTIONS
    """

    def add_region(
            self,
            name: str,
            data: DFType,
            manager: StrDict = None
    ):
        """Adds a region-type layer to the builder.

        Region-type layers should consist of Polygon-like geometries.
        Best results are read from a GeoDataFrame, or DataFrame.

        :param name: The name this layer is to be stored with
        :type name: str
        :param data: The location of the data for this layer
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param manager: The plotly properties for this layer.
        :type manager: StrDict
        """
        _check_name(name, LayerType.REGION)
        data = _read_data_full(name, LayerType.REGION, data, allow_builtin=True)
        layer = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='RGN', VTYPE=data.VTYPE)
        data = data[['value_field', 'geometry']]
        layer['data'], layer['odata'] = data, data.copy(deep=True)
        if self.use_templates:
            layer['manager'] = deepcopy(get_template('region'))
        else:
            layer['manager'] = {}
        _update_manager(layer, **(manager or {}))
        self._get_regions()[name] = layer

    def _get_region(self, name: str) -> StrDict:
        """Retrieves a region layer from the builder.

        Internal version.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        try:
            return self._get_regions()[name]
        except KeyError:
            raise NoLayerError(name, LayerType.REGION)

    def get_region(self, name: str) -> StrDict:
        """Retrieves a region layer from the builder.

        External version, returns a deepcopy.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        return deepcopy(self._get_region(name))

    def _get_regions(self) -> Dict[str, StrDict]:
        """Retrieves the region layers from the builder.

        Internal version.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return self._container['regions']

    def get_regions(self) -> Dict[str, StrDict]:
        """Retrieves the region layers from the builder.

        External version, returns a deepcopy.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_regions())

    def remove_region(self, name: str, pop: bool = False) -> StrDict:
        """Removes a region layer from the builder.

        :param name: The name of the layer to remove
        :type name: str
        :param pop: Whether to return the removed layer or not
        :type pop: bool
        :return: The removed layer (pop=True)
        :rtype: StrDict
        """
        try:
            region = self._get_regions().pop(name)
            if pop:
                return region
        except KeyError:
            raise NoLayerError(name, LayerType.REGION)

    def update_region_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a region or regions.

        The manager consists of Plotly properties.
        If the given name is none, all region layers will be updated.

        :param name: The name of the layer to update
        :type name: str
        :param updates: A dict containing updates for the layer(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the layer(s)
        :type kwargs: **kwargs
        """

        if name is None:
            for _, v in self._get_regions().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_region(name), updates=updates, overwrite=overwrite, **kwargs)

    def clear_region_manager(self, name: str = None):
        """Clears the manager of a region layer.

        If the given name is none, clears all of the region managers.
        """
        self.update_region_manager(name=name, updates={}, overwrite=True)

    def reset_region_data(self, name: str = None):
        """Resets the data within the region layer to the data that was input at the beginning.

        If the given name is None, all region layers will be reset.

        :param name: The name of the layer to reset
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
            longitude_field: str = None,
            convex_simplify: bool = False
    ):
        """Adds a grid-type layer to the builder.

        Grid-type layers should consist of Polygon-like or Point-like geometries.

        :param name: The name this layer is to be stored with
        :type name: str
        :param data: The location of the data for this layer
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param hex_resolution: The hexagonal resolution to use for this layer (None->builder default)
        :type hex_resolution: int
        :param latitude_field: The latitude column within the data
        :type latitude_field: str
        :param longitude_field: The longitude column within the data
        :type longitude_field: str
        :param convex_simplify: Determines if the area the grid is to be placed over should be simplified or not
        :type convex_simplify: bool
        """
        if hex_resolution is None:
            try:
                hex_resolution = self._get_hexbin()['HRES']
            except NoLayerError:
                pass

        selected_res = hex_resolution or self.default_hex_resolution

        _check_name(name, LayerType.GRID)
        data = _read_data_full(name, LayerType.GRID, data, allow_builtin=True)
        layer = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='GRD', VTYPE=data.VTYPE, HRES=selected_res)
        data = _convert_latlong_data(name, LayerType.GRID, data,
                                     latitude_field=latitude_field, longitude_field=longitude_field)
        if convex_simplify:
            data = GeoDataFrame(geometry=data.convex_hull, crs="EPSG:4326")

        data = _convert_to_hexbin_data(
            name, LayerType.GRID,
            data,
            hex_resolution=selected_res,
            binning_fn=lambda lst: 0
        )
        data = data[['value_field', 'geometry']]
        layer['data'], layer['odata'] = data, data.copy(deep=True)
        layer['manager'] = self._grid_manager
        self._get_grids()[name] = layer

    def _get_grid(self, name: str) -> StrDict:
        """Retrieves a grid layer from the builder.

        Internal version.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        try:
            return self._get_grids()[name]
        except KeyError:
            raise NoLayerError(name, LayerType.GRID)

    def get_grid(self, name: str) -> StrDict:
        """Retrieves a grid layer from the builder.

        External version, returns a deepcopy.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        return deepcopy(self._get_grid(name))

    def _get_grids(self) -> Dict[str, StrDict]:
        """Retrieves the grid layers from the builder.

        Internal version.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return self._container['grids']

    def get_grids(self) -> Dict[str, StrDict]:
        """Retrieves the grid layers from the builder.

        External version, returns a deepcopy.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_grids())

    def remove_grid(self, name: str, pop: bool = False) -> StrDict:
        """Removes a grid layer from the builder.

        :param name: The name of the layer to remove
        :type name:
        :param pop: Whether to return the removed layer or not
        :type pop: bool
        :return: The removed layer (pop=True)
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
            raise NoLayerError(name, LayerType.GRID)

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
        """Clears the manager of a region layer.

        If the given name is none, clears all of the region managers.
        """
        if self._get_grids():
            self._grid_manager.clear()

    def reset_grid_data(self, name: str = None):
        """Resets the data within the grid layer to the data that was input at the beginning.

        If the given name is None, all grid layers will be reset.

        :param name: The name of the layer to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_grids().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_grid(name))

    def reset_grids(self):
        """Resets the grid layer container to it's original state.
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
        """Adds a outline-type layer to the builder.

        :param name: The name this layer is to be stored with
        :type name: str
        :param data: The location of the data for this layer
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param as_boundary: Changes the data into one big boundary if true
        :type as_boundary: bool
        :param manager: Plotly properties for this layer
        :type manager: StrDict
        """
        _check_name(name, LayerType.OUTLINE)
        data = _read_data_full(name, LayerType.OUTLINE, data, allow_builtin=True)
        layer = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='OUT', VTYPE=data.VTYPE)
        data = _convert_latlong_data(name, LayerType.OUTLINE, data, latitude_field=latitude_field,
                                     longitude_field=longitude_field)[['value_field', 'geometry']]
        # TODO: there exists errors with the representation of outlines (given lfields)

        if as_boundary:
            data = gcg.unify_geodataframe(data)

        layer['data'], layer['odata'] = data, data.copy(deep=True)
        if self.use_templates:
            layer['manager'] = deepcopy(get_template('outline'))
        else:
            layer['manager'] = {}
        _update_manager(layer, **(manager or {}))
        self._get_outlines()[name] = layer

    def _get_outline(self, name: str) -> StrDict:
        """Retrieves a outline layer from the builder.

        Internal version.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        try:
            return self._get_outlines()[name]
        except KeyError:
            raise NoLayerError(name, LayerType.OUTLINE)

    def get_outline(self, name: str) -> StrDict:
        """Retrieves a outline layer from the builder.

        External version, returns a deepcopy.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        return deepcopy(self._get_outline(name))

    def _get_outlines(self) -> Dict[str, StrDict]:
        """Retrieves the outline layers from the builder.

        Internal version.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return self._container['outlines']

    def get_outlines(self) -> Dict[str, StrDict]:
        """Retrieves the outline layers from the builder.

        External version, returns a deepcopy.

        :return: The retrieved layers
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_outlines())

    def remove_outline(self, name: str, pop: bool = False) -> StrDict:
        """Removes an outline layer from the builder.

        :param name: The name of the layer to remove
        :type name:
        :param pop: Whether to return the removed layer or not
        :type pop: bool
        :return: The removed layer (pop=True)
        :rtype: StrDict
        """
        try:
            outline = self._get_outlines().pop(name)
            if pop:
                return outline
        except KeyError:
            raise NoLayerError(name, LayerType.OUTLINE)

    def update_outline_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a outline or outlines.

        The manager consists of Plotly properties.
        If the given name is none, all outline layers will be updated.

        :param name: The name of the layer to update
        :type name: str
        :param updates: A dict containing updates for the layer(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the layer(s)
        :type kwargs: **kwargs
        """

        if name is None:
            for _, v in self._get_outlines().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_outline(name), updates=updates, overwrite=overwrite, **kwargs)

    def clear_outline_manager(self, name: str = None):
        """Clears the manager of a outline layer.

        If the given name is none, clears all of the outline managers.
        """
        self.update_outline_manager(name=name, updates={}, overwrite=True)

    def reset_outline_data(self, name: str = None):
        """Resets the data within the outline layer to the data that was input at the beginning.

        If the given name is None, all outline layers will be reset.

        :param name: The name of the layer to reset
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
            text_field: str = None,
            manager: StrDict = None
    ):
        """Adds a point-type layer to the builder.

        Ideally the layer's 'data' member should contain
        lat/long columns or point like geometry column. If the geometry column
        is present and contains no point like geometry, the geometry will be converted
        into a bunch of points.

        :param name: The name this layer is to be stored with
        :type name: str
        :param data: The location of the data for this layer
        :type data: Union[str, DataFrame, GeoDataFrame]
        :param latitude_field: The latitude column of the data
        :type latitude_field: str
        :param longitude_field: The longitude column of the data
        :type longitude_field: str
        :param text_field: The column containing text for data entries
        :type text_field: str
        :param manager: Plotly properties for this layer
        :type manager: StrDict
        """

        _check_name(name, LayerType.POINT)
        data = _read_data_full(name, LayerType.POINT, data, allow_builtin=False)
        layer = dict(NAME=name, RTYPE=data.RTYPE, DSTYPE='PNT', VTYPE=data.VTYPE)
        if text_field:
            data = _convert_latlong_data(name, LayerType.POINT, data, latitude_field=latitude_field,
                                         longitude_field=longitude_field)[[text_field, 'value_field', 'geometry']]
        else:
            data = _convert_latlong_data(name, LayerType.POINT, data, latitude_field=latitude_field,
                                         longitude_field=longitude_field)[['value_field', 'geometry']]

        layer['data'], layer['odata'] = data, data.copy(deep=True)
        layer['tfield'] = text_field
        if self.use_templates:
            layer['manager'] = deepcopy(get_template('point'))
        else:
            layer['manager'] = {}

        _update_manager(layer, **(manager or {}))
        self._get_points()[name] = layer

    def _get_point(self, name: str) -> StrDict:
        """Retrieves a point layer from the builder.

        Internal version.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        try:
            return self._get_points()[name]
        except KeyError:
            raise NoLayerError(name, LayerType.POINT)

    def get_point(self, name: str) -> StrDict:
        """Retrieves a point layer from the builder.

        External version, returns a deepcopy.

        :param name: The name of the layer
        :type name: str
        :return: The retrieved layer
        :rtype: StrDict
        """
        return deepcopy(self._get_point(name))

    def _get_points(self) -> Dict[str, StrDict]:
        """Retrieves the collection of point layers in the builder.

        Internal version.

        :return: The point layers within the builder
        :rtype: Dict[str, StrDict]
        """
        return self._container['points']

    def get_points(self) -> Dict[str, StrDict]:
        """Retrieves the collection of point layers in the builder.

        External version, returns a deepcopy.

        :return: The point layers within the builder
        :rtype: Dict[str, StrDict]
        """
        return deepcopy(self._get_points())

    def remove_point(self, name: str, pop: bool = False) -> StrDict:
        """Removes a point layer from the builder.

        :param name: The name of the layer to remove
        :type name: str
        :param pop: Whether to return the removed layer or not
        :type pop: bool
        :return: The removed layer (pop=True)
        :rtype: StrDict
        """
        try:
            point = self._get_points().pop(name)
            if pop:
                return point
        except KeyError:
            raise NoLayerError(name, LayerType.POINT)

    def update_point_manager(self, name: str = None, updates: StrDict = None, overwrite: bool = False, **kwargs):
        """Updates the manager of a point or points.

        The manager consists of Plotly properties.
        If the given name is none, all point layers will be updated.

        :param name: The name of the layer to update
        :type name: str
        :param updates: A dict containing updates for the layer(s)
        :type updates: StrDict
        :param overwrite: Whether to override the current properties with the new ones or not
        :type overwrite: bool
        :param kwargs: Other updates for the layer(s)
        :type kwargs: **kwargs
        """

        if name is None:
            for _, v in self._get_points().items():
                _update_manager(v, updates=updates, overwrite=overwrite, **kwargs)
        else:
            _update_manager(self._get_point(name), updates=updates, overwrite=overwrite, **kwargs)

    def clear_point_manager(self, name: str = None):
        """Clears the manager of a point layer.

        If the given name is none, clears all of the point managers.
        """
        self.update_point_manager(name=name, updates={}, overwrite=True)

    def reset_point_data(self, name: str = None):
        """Resets the data within the point layer to the data that was input at the beginning.

        If the given name is None, all point layers will be reset.

        :param name: The name of the layer to reset
        :type name: str
        """
        if name is None:
            for _, v in self._get_points().items():
                _reset_to_odata(v)
        else:
            _reset_to_odata(self._get_point(name))

    def reset_points(self):
        """Resets the point layer container to its original state.
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
        """Applies a function to the layers within a query.

        For advanced users and not to be used carelessly.
        The functions first argument must be the layer.

        :param name: The query of the layers to apply the function to
        :type name: str
        :param fn: The function to apply
        :type fn: Callable
        :param allow_empty: Whether to allow query arguments that retrieved empty results or not
        :type allow_empty: bool
        """

        layers = self._search(name)
        lst = []

        if not allow_empty and not layers:
            raise ValueError("The query submitted returned an empty result.")
        if 'data' in layers:
            lst.append(fn(layers, *args, **kwargs))
        else:
            for _, v in layers.items():
                if not allow_empty and not v:
                    raise ValueError("The query submitted returned an empty result.")
                if 'data' in v:
                    lst.append(fn(v, *args, **kwargs))
                else:
                    for _, vv in v.items():
                        if not allow_empty and not vv:
                            raise ValueError("The query submitted returned an empty result.")
                        if 'data' in vv:
                            lst.append(fn(vv, *args, **kwargs))
                        else:
                            for _, vvv in vv.items():
                                if not allow_empty and not vvv:
                                    raise ValueError("The query submitted returned an empty result.")
                                if 'data' in vvv:
                                    lst.append(fn(vvv, *args, **kwargs))
                                else:
                                    raise ValueError("Error when applying function to query.")
        return lst

    def remove_empties(self, empty_symbol: Any = 0, add_to_plot: bool = True):
        """Removes empty entries from the hexbin layer.

        The empty entries may then be added to the plot as a grid.

        :param empty_symbol: The symbol that constitutes an empty value in the layer
        :type empty_symbol: Any
        :param add_to_plot: Whether to add the empty cells to the plot or not
        :type add_to_plot: bool
        """
        layer = self._get_hexbin()

        if add_to_plot:
            empties = layer['data'][layer['data']['value_field'] == empty_symbol]
            if not empties.empty:
                empties['value_field'] = 0
                self.get_grids()['|*EMPTY*|'] = empties
        layer['data'] = layer['data'][layer['data']['value_field'] != empty_symbol]

    # this is both a data altering, and plot altering function
    def logify_scale(self, **kwargs):
        """Makes the scale of the hexbin layers logarithmic.

        This function changes the tick values and tick text of the scale.
        The numerical values on the scale are the exponent of the tick text,
        i.e the text of 1 on the scale actually represents the value of zero,
        and the text of 1000 on the scale actually represents the value of 3.

        :param kwargs: Keyword arguments to be passed into logify functions
        :type kwargs: **kwargs
        """

        layer = self._get_hexbin()

        if layer['VTYPE'] == 'STR':
            raise TypeError("The scale of a hexbin layer that is binned "
                            "based on qualitative data can not be converted into a logarithmic scale.")
        _update_manager(layer, butil.logify_scale(layer['data'], **kwargs))

    def clip_layers(self, clip: str, to: str, method: str = 'sjoin', reduce_first: bool = True,
                      operation: str = 'intersects'):
        """Clips a query of layers to another layer.
        this function is experimental and may not always work as intended

        There are two methods for this clipping:
        1) sjoin -> Uses GeoPandas spatial join in order to clip geometries
                    that (intersect, are within, contain, etc.) the geometries
                    acting as the clip.
        2) gpd  ->  Uses GeoPandas clip function in order to clip geometries
                    to the boundary of the geometries acting as the clip.

        :param clip: The query for the layers that are to be clipped to another
        :type clip: GeoDataFrame
        :param to: The query for the layers that are to be used as the boundary
        :type to: GeoDataFrame
        :param method: The method to use when clipping, one of 'sjoin', 'gpd'
        :type method: str
        :param reduce_first: Determines whether the geometries acting as the clip should be reduced first or not
        :type reduce_first: bool
        :param operation: The operation to apply when using sjoin (spatial join operation)
        :type operation: str
        """

        # TODO: In the future combining the geometries of all dataframes may be useful. (geopandas overlay how=union)
        datas = self.apply_to_query(to, lambda layer: layer['data'])
        if reduce_first:
            datas = [GeoDataFrame(geometry=[reduce(lambda left, right: left.union(right), datas).unary_union],
                                  crs="EPSG:4326")]

        def gpdhelp(layer: dict):
            layer['data'] = butil.gpd_clip(layer['data'], datas, validate=True)

        def sjoinhelp(layer: dict):
            layer['data'] = butil.sjoin_clip(layer['data'], datas, operation=operation, validate=True)

        if method == 'gpd':
            self.apply_to_query(clip, gpdhelp)
        elif method == 'sjoin':
            self.apply_to_query(clip, sjoinhelp)
        else:
            raise ValueError("When clipping layers, the selected method must be one of ['gpd', 'sjoin'].")

    # check methods of clipping
    def simple_clip(self, method: str = 'sjoin'):
        """Quick general clipping.

        This function clips the hexbin layer and grid layers to the region and outline layers.
        The function also clips the point layers to the hexbin, region, grid, and outline layers.

        :param method: The method to use when clipping, one of 'sjoin' or 'gpd'
        :type method: str
        """

        self.clip_layers('hexbin+grids', 'regions+outlines', method=method, operation='intersects')
        # self.clip_layers('hexbin+grids', 'outlines', method=method, operation='intersects')
        self.clip_layers('points', 'hexbin+regions+outlines+grids', method=method, operation='within')

    def _remove_underlying_grid(self, df: GeoDataFrame, gdf: GeoDataFrame):
        """Removes the pieces of a GeoDataFrame that lie under another.

        :param df: The overlayed dataframe (after alteration)
        :type df: GeoDataFrame
        :param gdf: The merged underlying dataframe (after alteration)
        :type gdf: GeoDataFrame
        """

        if not df.empty and not gdf.empty:
            df = df[['value_field', 'geometry']]
            gdf = gpd.overlay(gdf, df[df['value_field'].notnull()],
                              how='difference')
            gdf = gcg.remove_other_geometries(gdf, 'Polygon')
            return gdf

    """
    PLOT ALTERING FUNCTIONS
    """

    def adjust_figure(self, width=600, height=450):

        self._figure.update_traces(patch=dict(colorbar_ypad=0), selector=dict(type='choropleth'))
        self._figure.update_layout(height=height, width=width, margin=dict(r=90, l=0, t=10, b=10), overwrite=True)

    def adjust_colorbar_size(self, width=700, height=450, t=20, b=20):
        """Adjusts the color scale position of the color bar to match the plot area size.

        Does not work.
        """
        return

    def adjust_opacity(self, alpha: float = None):
        """Conforms the opacity of the color bar of the hexbin layer to an alpha value.

        The alpha value can be passed in as a parameter, otherwise it is taken
        from the marker.opacity property within the layer's manager.

        :param alpha: The alpha value to conform the color scale to
        :type alpha: float
        """

        layer = self._get_hexbin()
        butil.opacify_colorscale(layer, alpha=alpha)

    def adjust_focus(self, on: str = 'hexbin', center_on: bool = False, rotation_on: bool = True,
                     ranges_on: bool = True, rot_buffer_lat: float = 0, rot_buffer_lon: float = 0,
                     buffer_lat: tuple = (0, 0), buffer_lon: tuple = (0, 0)):
        """Focuses on layer(s) within the plot.

        Collects the geometries of the queried layers in order to
        obtain a boundary to focus on.

        In the future using a GeoSeries may be looked into for cleaner code.

        :param on: The query for the layer(s) to be focused on
        :type on: str
        :param center_on: Whether or not to add a center component to the focus
        :type center_on: bool
        :param rotation_on: Whether or not to add a projection rotation to the focus
        :type rotation_on: bool
        :param ranges_on: Whether or not to add a lat axis, lon axis ranges to the focus
        :type ranges_on: bool
        :param rot_buffer_lat: A number to add or subtract from the automatically calculated latitude (rotation)
        :type rot_buffer_lat: float
        :param rot_buffer_lon: A number to add or subtract from the automatically calculated longitude (rotation)
        :type rot_buffer_lon: float
        :param buffer_lat: A low and high bound to add and subtract from the lataxis range
        :type buffer_lat: Tuple[float, float]
        :param buffer_lon: A low and high bound to add and subtract from the lonaxis range
        :type buffer_lon: Tuple[float, float]
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

        center = gcg.find_center_simple(geoms)
        center = dict(lon=center.x, lat=center.y)

        if rotation_on:
            geos['projection_rotation'] = dict(lat=center['lat'] + rot_buffer_lat, lon=center['lon'] + rot_buffer_lon)

        if center_on:
            geos['center'] = center

        self._figure.update_geos(**geos)

    def auto_grid(self, on: str = 'hexbin', by_bounds: bool = False, hex_resolution: int = None):
        """Makes a grid over the queried layers.

        :param on: The query for the layers to have a grid generated over them
        :type on: str
        :param by_bounds: Whether or not to treat the  geometries as a single boundary
        :type by_bounds: bool
        :param hex_resolution: The hexagonal resolution to use for the auto grid
        :type hex_resolution: int
        """

        fn = gcg.generate_grid_over if by_bounds else gcg.hexify_dataframe
        hex_resolution = hex_resolution if hex_resolution is not None else self.default_hex_resolution

        def helper(layer):
            return fn(layer['data'], hex_resolution=hex_resolution)

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
        """Converts the color scale of the layer(s) to a discrete scale.

        :param scale_type: One of 'sequential', 'discrete' for the type of color scale being used
        :type scale_type: str
        :param kwargs: Keyword arguments to be passed into the discretize functions
        :type kwargs: **kwargs
        """

        layer = self._get_hexbin()

        if layer['VTYPE'] == 'STR':
            raise ValueError(f"You can not discretize a qualitative layer.")
        else:
            low = layer['manager'].get('zmin', min(layer['data']['value_field']))
            high = layer['manager'].get('zmax', max(layer['data']['value_field']))
            layer['manager']['colorscale'] = discretize_cscale(layer['manager'].get('colorscale'), scale_type,
                                                                 low, high, **kwargs)

    """
    RETRIEVAL/SEARCHING FUNCTIONS
    
    get_regions(), etc... could also fall under here.
    """

    def search(self, query: str) -> StrDict:
        """Query the builder for specific layer(s).

        Each query argument should be formatted like:
        <regions|grids|outlines|points|main|hexbin|all>
        OR
        <region|grid|outline|point>:<name>

        And each query argument can be separated by the '+' character.

        External version.

        :param query: The identifiers for the layers being searched for
        :type query: str
        :return: The result of the query
        :rtype: StrDict
        """
        return deepcopy(self._search(query))

    def _search(self, query: str, big_query: bool = True, multi_query: bool = True) -> StrDict:
        """Query the builder for specific layer(s).

        Internal version, see search().

        :param query: The identifiers for the layers being searched for
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
        """Query the builder for specific layers()

        Retrieves a query of a single argument only, see _search().

        :param query: The identifier for the layer(s) being searched for
        :type query: str
        :param big_query: Whether to allow the query argument to represent a collection of layers or not
        :type big_query: bool
        :return: The result of the query
        :rtype: StrDict
        """
        type_ret = dict(region=self._get_region, grid=self._get_grid, outline=self.get_outline, point=self._get_point)

        if query in ['main', 'hexbin']:
            return self._get_hexbin()
        elif query in ['regions', 'grids', 'outlines', 'points', 'all']:
            if big_query:
                return self._container if query == 'all' else self._container[query]
            else:
                raise BuilderQueryInvalidError("The given query should not refer to a collection of layers.")

        try:
            typer, name = _split_query(query)
        except ValueError:
            raise BuilderQueryInvalidError(
                "The given query should be one of ['regions', 'grids', 'outlines', 'points', "
                f"'main', 'hexbin'] or in the form of '<type>:<name>'.\nReceived: {query}.")

        try:
            return type_ret[typer](name)
        except KeyError:
            raise BuilderQueryInvalidError("The given layer type does not exist.\nMust be one of ['region', "
                                           f"'grid', 'outline', 'point'].\n Received: {typer}")

    get_query_data = lambda self, name: self.apply_to_query(name, lambda ds: ds['data'])

    """
    PLOTTING FUNCTIONS
    """

    def plot_regions(self):
        """Plots the region layers within the builder.

        All of the regions are treated as separate plot traces.
        """
        # logger.debug('adding regions to plot.')
        mapbox = self.plot_output_service == 'mapbox'
        if not self._get_regions():
            raise NoLayersError(LayerType.REGION)
        for regname, regds in self._get_regions().items():
            self._figure.add_trace(_prepare_choropleth_trace(
                gcg.conform_geogeometry(regds['data']),
                mapbox=mapbox).update(regds['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_grids(self, remove_underlying: bool = False):
        """Plots the grid layers within the builder.

        Merges all of the layers together, and plots it as a single plot trace.
        """
        if not self._get_grids():
            raise NoLayersError(LayerType.GRID)

        merged = gcg.conform_geogeometry(
            pd.concat(self.apply_to_query('grids', lambda layer: layer['data'])).drop_duplicates())

        if remove_underlying:
            merged = self._remove_underlying_grid(self._get_hexbin()['data'], merged)
        self._figure.add_trace(_prepare_choropleth_trace(
            merged,
            mapbox=self.plot_output_service == 'mapbox').update(text='GRID').update(self._grid_manager))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_hexbin(self):
        """Plots the hexbin layer within the builder.

        If qualitative, the layer is split into uniquely labelled plot traces.
        """

        layer = self._get_hexbin()
        df = gcg.conform_geogeometry(layer['data'])

        # qualitative layer
        if layer['VTYPE'] == 'STR':
            df['text'] = 'BEST OPTION: ' + df['value_field']
            colorscale = layer['manager'].pop('colorscale')
            try:
                colorscale = get_scale(colorscale)
            except AttributeError:
                pass

            sep = {}
            mapbox = self.plot_output_service == 'mapbox'

            df['temp_value'] = df['value_field']
            df['value_field'] = 0

            for i, name in enumerate(df['temp_value'].unique()):
                sep[name] = df[df['temp_value'] == name]

            manager = deepcopy(layer['manager'])
            if isinstance(colorscale, dict):
                for k, v in sep.items():
                    try:
                        manager['colorscale'] = solid_scale(colorscale[k])
                    except KeyError:
                        raise ColorScaleError(f"If the colorscale is a map, you must provide hues for each option.\n"
                                              f"There was no color specified for the unique option of '{k}'.")
                    self._figure.add_trace(_prepare_choropleth_trace(
                        v,
                        mapbox=mapbox).update(name=k, showscale=False, showlegend=True, text=v['text']).update(manager))

            elif isinstance(colorscale, list) or isinstance(colorscale, tuple):

                for i, (k, v) in enumerate(sep.items()):
                    try:
                        if isinstance(colorscale[i], list) or isinstance(colorscale[i], tuple) \
                                or isinstance(colorscale[i], set):
                            manager['colorscale'] = solid_scale(colorscale[i][1])
                        else:
                            manager['colorscale'] = solid_scale(colorscale[i])
                    except IndexError:
                        raise ColorScaleError(f"There were not enough hues for all of the "
                                              f"unique options in the layer.\n"
                                              f"There was no color specified for the unique option of '{k}'.\n"
                                              f"# of colors supplied: {len(colorscale)}\n"
                                              f"# of colors needed: {len(sep.items())}")
                    self._figure.add_trace(
                        _prepare_choropleth_trace(v, mapbox=mapbox).update(name=k, showscale=False, showlegend=True,
                                                                           text=v['text']).update(manager))
            else:
                raise ColorScaleError("The color scale given is not valid.\n "
                                      "The color scale must be one of the following options:\n"
                                      "1) a valid Plotly color scale (string);\n"
                                      "2) a dictionary-style collection (including hues for all unique options); or\n"
                                      "3) a list-style collection (including hues for all unique options).")
        # quantitative layer
        else:
            df['text'] = 'VALUE: ' + df['value_field'].astype(str)
            self._figure.add_trace(_prepare_choropleth_trace(df, mapbox=self.plot_output_service == 'mapbox').update(
                text=df['text']).update(layer['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def plot_outlines(self, raise_errors: bool = False):
        """Plots the outline layers within the builder.

        All of the outlines are treated as separate plot traces.
        The layers must first be converted into point-like geometries.

        :param raise_errors: Whether or not to throw errors upon reaching empty dataframes
        :type raise_errors: bool
        """
        if not self._get_outlines():
            raise NoLayersError(LayerType.OUTLINE)

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
        """Plots the point layers within the builder.

        All of the point are treated as separate plot traces.
        """

        if not self._get_points():
            raise NoLayersError(LayerType.POINT)

        mapbox = self.plot_output_service == 'mapbox'
        for poiname, poids in self._get_points().items():
            self._figure.add_trace(_prepare_scattergeo_trace(
                poids['data'],
                separate=False,
                disjoint=False,
                mapbox=mapbox).update(name=poiname,
                                      text=poids['data'][poids['tfield']] if poids['tfield'] else None)
                                   .update(poids['manager']))

        self._plot_status = PlotStatus.DATA_PRESENT

    def set_mapbox(self, accesstoken: str):
        """Prepares the builder for a mapbox output.

        Sets figure.layout.mapbox_accesstoken, and plot_settings output service.

        :param accesstoken: A mapbox access token for the plot
        :type accesstoken: str
        """
        self.plot_output_service = 'mapbox'
        self._figure.update_layout(mapbox_accesstoken=accesstoken)

    def finalize(
            self,
            plot_regions: bool = True,
            plot_grids: bool = True,
            plot_hexbin: bool = True,
            plot_outlines: bool = True,
            plot_points: bool = True,
            raise_errors: bool = False
    ):
        """Builds the final plot by adding traces in order.

        Invokes the functions in the following order:
        1) plot regions
        2) plot grids
        3) plot layer
        4) plot outlines
        5) plot points

        In the future we should alter these functions to
        allow trace order implementation.

        :param plot_regions: Whether or not to plot region layers
        :type plot_regions: bool
        :param plot_grids: Whether or not to plot grid layers
        :type plot_grids: bool
        :param plot_hexbin: Whether or not to plot the hexbin layer
        :type plot_hexbin: bool
        :param plot_outlines: Whether or not to plot outline layers
        :type plot_outlines: bool
        :param plot_points: Whether or not to plot point layers
        :type plot_points: bool
        :param raise_errors: Whether or not to raise errors related to empty layer collections
        :type raise_errors: bool
        """
        if plot_regions:
            try:
                self.plot_regions()
            except NoLayersError as e:
                if raise_errors:
                    raise e from None
        if plot_grids:
            try:
                self.plot_grids(remove_underlying=True)
            except NoLayersError as e:
                if raise_errors:
                    raise e from None
        if plot_hexbin:
            try:
                self.plot_hexbin()
            except NoLayerError as e:
                if raise_errors:
                    raise e from None
        if plot_outlines:
            try:
                self.plot_outlines()
            except NoLayersError as e:
                if raise_errors:
                    raise e from None
        if plot_points:
            try:
                self.plot_points()
            except NoLayersError as e:
                if raise_errors:
                    raise e from None

    def output(self, filepath: str = None, clear_figure: bool = False,
               crop_output: bool = False, percent_retain=None, keep_original: bool = False, **kwargs):
        """Outputs the figure to a filepath.

        The figure is output via Plotly's write_image() function.
        Plotly's Kaleido is required for this feature.

        :param filepath: The filepath to output the figure at (including filename and extension)
        :type filepath: str
        :param clear_figure: Whether or not to clear the figure after this operation
        :type clear_figure: bool
        :param crop_output: Whether or not to crop the output figure (requires that extension be pdf, PdfCropMargins must be installed)
        :type crop_output: bool
        :param percent_retain: Percentage of margins to retain from crop (requires that extension be pdf, PdfCropMargins must be installed)
        :type percent_retain: float, str, list
        :param keep_original: Whether or not to keep the original figure (requires that extension be pdf, PdfCropMargins must be installed)
        :type keep_original: bool
        :param kwargs: Keyword arguments for the write_image function
        :type kwargs: **kwargs
        """
        if filepath is None:
            if self.output_destination is None:
                raise NoFilepathError("There must either be a filepath given directly or one stored in"
                                      " the builder's 'output_destination' property.")
            filepath = self.output_destination

        self._figure.write_image(filepath, **kwargs)
        self._last_output_location = filepath
        if clear_figure:
            self.clear_figure()

        if crop_output:
            from pdfCropMargins import crop
            newfilepath, ext = os.path.splitext(filepath)
            newfilepath = f"{newfilepath}-cropped{ext}"

            crop_args = []
            if percent_retain is not None:
                if isinstance(percent_retain, list):
                    crop_args.append("-p4")
                    crop_args.extend(percent_retain)
                else:
                    crop_args.extend(["-p", percent_retain])
            else:
                crop_args.extend(["-p4", "1.5", "4.5", "2.5", "1.5"])
            crop_args.extend(["-u", "-s", filepath, "-o", newfilepath])
            # print(f"Call to PdfCropMargins: {crop_args}")
            crop(crop_args)

            if not keep_original:
                os.remove(filepath)
                os.rename(newfilepath, filepath)

    def display(self, clear_figure: bool = False, **kwargs):
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

        if self.use_templates:
            self._figure.update(get_template('figure'))
            # grids will all reference this manager
            self._grid_manager = deepcopy(get_template('grid'))
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
        """Resets the layers of the builder to their original state.
        """
        try:
            self.reset_hexbin_data()
        except NoLayerError:
            pass
        self.reset_region_data()
        self.reset_grid_data()
        self.reset_outline_data()
        self.reset_point_data()

    @staticmethod
    def builder_from_dict(builder_dict: StrDict = None, **kwargs):
        """Makes a builder from a dictionary.

        :param builder_dict: The dictionary to build from
        :type builder_dict: StrDict
        :param kwargs: Keyword arguments for the builder
        :type kwargs: **kwargs
        """
        settings = simplify_dicts(builder_dict, **kwargs)
        figure_manager = settings.pop('figure_manager', {})
        figure_manager.update(dict(layout=settings.pop('figure_layout', {}), geos=settings.pop('figure_geos', {})))
        grid_manager = settings.pop('grid_manager', {})

        builder = PlotBuilder(**settings)
        builder.update_grid_manager(grid_manager)
        builder.update_figure(figure_manager)
        return builder
