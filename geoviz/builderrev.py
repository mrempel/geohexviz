from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly
from pandas import Series
from plotly.graph_objects import Figure, Choropleth, Scattergeo
from functools import reduce

from .utils import geoutils as gcg
from . import errors as gce
from .utils.util import dict_deep_update, fix_filepath, get_sorted_occurrences, \
    make_multi_dataset, dissolve_multi_dataset, get_stats, rename_dataset, get_column_or_default, \
    generate_dataframe_random_ids, get_column_type
from .utils import plot_util as butil
from .utils.colorscales import solidScales, getDiscreteScale, getScale, tryGetScale, configureScaleWithAlpha

import warnings
import os
from os.path import join as pjoin
import sys
from typing import List, Dict, Any, Optional, ClassVar, Union, Tuple
from shapely.geometry import Polygon, Point, MultiPolygon
from .structures.tracecontainer import PlotlyDataContainer

import logging
from copy import deepcopy

pd.options.mode.chained_assignment = None
assume_changes = True

_group_functions = {
    'sum': lambda lst: sum(lst),
    'avg': lambda lst: sum(lst) / len(lst),
    'min': lambda lst: min(lst),
    'max': lambda lst: max(lst),
    'count': lambda lst: len(lst),
    'bestworst': get_sorted_occurrences
}

DataSet = Dict[str, Any]
DataSets = Dict[str, DataSet]
DataSetManager = Dict[str, Any]
FigureManager = Dict[str, Any]
FileOutputManager = Dict[str, Any]
DataFrame = pd.DataFrame
GeoDataFrame = gpd.GeoDataFrame

# pio.kaleido.scope.topojson = pjoin(pjoin(pjoin(pjoin(__file__, 'data'), 'envplotly'), 'plotly-topojson'), '')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
Notes for refactoring:

Think about the overall process, can data be clipped when it is passed in?
How can we rearrange the process to make the steps more and more like:

1) Manipulate data
1.5) Get Plotly properties
2) Plot data

And hence we may be able to separate it from Plotly entirely, as they are two separate things.
"""


class FocusMode(Enum):
    """Defines the different focus modes that the builder has.
    """
    AUTO_FITBOUND = 'auto-fitbound'
    AUTO_BUILDER = 'auto-builder'
    MANUAL_BUILDER = 'manual-builder'


def loggingAddFile(filepath):
    fh = logging.FileHandler(filepath, 'w+')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.addHandler(fh)
    fh.setFormatter(formatter)


def toggleLogging(enabled):
    logger.disabled = not enabled


def builder_from_dict(builder_dict: Dict[str, Any] = None, **kwargs):
    settings = {}
    if builder_dict:
        settings.update(builder_dict)
    settings.update(kwargs)

    plotly_managers = settings.pop('builder_managers', {})
    plot_settings = settings.pop('plot_settings', {})

    builder = PlotBuilder(**settings)
    builder.update_dataset_manager(plotly_managers.get('main_dataset', {}))
    builder.update_grid_manager(plotly_managers.get('grids', {}))
    builder.update_figure_manager(plotly_managers.get('figure', {}))

    for k, v in plotly_managers.get('regions', {}):
        builder.update_region_manager(v, name=k)
    for k, v in plotly_managers.get('outlines', {}):
        builder.update_outline_manager(v, name=k)
    for k, v in plotly_managers.get('points', {}):
        builder.update_point_manager(v, name=k)

    builder.update_file_output_manager(plotly_managers.get('file_output', {}))
    builder.update_plot_settings(**plot_settings)

    return builder


def isvalid_dataset(ds: DataSet):
    if ds is not None:
        if 'data' in ds:
            return isvalid_dataframe(ds['data'])
    return False


isvalid_dataframe = lambda df: True if df is not None and not df.empty else False


def _is_list_like(sequence):
    return isinstance(sequence, list) or isinstance(sequence, tuple)


_extension_mapping = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    '.shp': gpd.read_file,
    '': gpd.read_file
}


def _read_dataset(dataset: DataSet) -> DataSet:
    """Converts the 'data' of the given hex to a proper GeoDataFrame.

    :param dataset: The dataset that will be converted into proper format
    :type dataset: DataSet
    """
    if 'data' not in dataset:
        raise gce.BuilderDatasetInfoError(
            "There must be a 'data' field passed (either DataFrame, GeoDataFrame, of filepath).")

    df = dataset.pop('data')
    # attempt to read the file as a DataFrame or GeoDataFrame
    if isinstance(df, str) or isinstance(df, dict):
        # this means it could be a file path
        # attempt to read the file

        try:
            behaviour = df.pop('behaviour', 'geopandas').lower()
            method = df.pop('method', 'read_file').lower()
            ar = df.pop('args')
            kw = df.pop('kwargs', {})

            if method.startswith('read_'):
                if behaviour == 'geopandas':
                    method = getattr(gpd, method)
                elif behaviour == 'pandas':
                    method = getattr(pd, method)
                else:
                    raise TypeError(
                        f"You may only select the behaviour to be one of (geopandas, pandas), got={behaviour}.")
            else:
                raise TypeError(f"You may only use a method that begins with 'read', got={method}.")

            df = method(*ar, **kw)

        except AttributeError:

            filepath, extension = os.path.splitext(pjoin(os.path.dirname(__file__), df))
            filepath = fix_filepath(filepath, add_ext=extension)
            try:
                df = _extension_mapping[extension](filepath)
            except KeyError:
                logger.warning("The general file formats accepted by this application are (.csv, .shp). Be careful.")
    else:
        df = df.copy(deep=True)

    dataset['data'] = df
    rename_dataset(dataset)

    df = dataset['data']
    latitude_field = get_column_or_default(df, 'latitude_field')
    longitude_field = get_column_or_default(df, 'longitude_field')
    geometry_field = get_column_or_default(df, 'geometry_field')
    txt_field = get_column_or_default(df, 'text_field', default_val=np.nan)
    hov_field = dataset.get('hover_fields', {})
    df['text_field'] = txt_field
    dataset['hover_fields'] = hov_field

    if isinstance(df, GeoDataFrame):
        if geometry_field is not None:
            if latitude_field is not None:
                warnings.warn('Latitude field should not be passed with the dataframe if geometry field is also '
                              'passed. Assuming the geometry field takes precedence...')
            elif longitude_field is not None:
                warnings.warn('Longitude field should not be passed with the dataframe if geometry field is also '
                              'passed. Assuming the geometry field takes precedence...')
            df.geometry = df['geometry_field']

        elif latitude_field is not None and longitude_field is not None:
            if geometry_field is not None:
                raise gce.BuilderDatasetInfoError('Geometry field should not be passed along with longitude and '
                                                  'latitude fields.')
            df.geometry = gpd.points_from_xy(df['latitude_field'], df['longitude_field'])

    # try to parse the geometry from the dataset
    elif isinstance(df, DataFrame):
        try:
            df = gcg.convert_dataframe_coordinates_to_geodataframe(df, latitude_field='latitude_field',
                                                                   longitude_field='longitude_field')
        except KeyError:
            try:
                df = gcg.convert_dataframe_geometry_to_geodataframe(df, geometry_field='geometry_field')
            except KeyError:
                raise gce.BuilderDatasetInfoError("The builder could not parse a dataset's information.")

    if not df.crs:
        df.crs = 'EPSG:4326'
    df.to_crs('EPSG:4326', inplace=True)

    dataset['data'] = df

    if not isvalid_dataset(dataset):
        raise gce.BuilderDatasetInfoError("You input an invalid dataset!")

    logger.debug('dataset data converted into GeoDataFrame.')
    dataset['manager'] = dataset.get('manager', {})


def _hexbinify_dataset(dataset: DataSet, hex_resolution: int) -> DataSet:
    """Sets the main dataset of the builder.

    bins are dicts formatted as such:
    <name of bin> = {
        frame = <filename | GeoDataFrame | DataFrame>
        latitude_field = <lat column in dataframe> (Opt)
        longitude_field = <lon column in dataframe> (Opt)
        geometry_field = <geometry column in dataframe> (Opt)
        binning_field = <column to group the data by> (Opt)
        binning_fn = <the function to apply to the grouped data (lambda)> (Opt)
        value_field = <column containing pre-existing values to plot> (Opt)
        manager = <dict of management properties for the plotting of this bin> (Opt)
    }

    Keep in mind only certain things in the above example should be present when
    the user inputs their dict.

    :param name: The name this dataset will be referred to as
    :type name: str
    :param hexbin: A bin as shown above
    :param hexbin: A bin as shown above
    :type hexbin: DataSet
    :param to_point: Determines if the point data of the bins should be plotted as well
    :type to_point: bool
    """

    hex_resolution = dataset.get('hex_resolution', hex_resolution)
    dataset['binning_fn'] = dataset.get('binning_fn', None)
    binning_kw = {}
    binning_args = ()
    if isinstance(dataset['binning_fn'], dict):
        bfn = dataset['binning_fn'].pop('fn')
        if 'args' in dataset['binning_fn']:
            binning_args = dataset['binning_fn'].pop('args', tuple())
            binning_kw = dataset['binning_fn'].pop('kwargs', dict())
        else:
            binning_kw = dataset['binning_fn']
        dataset['binning_fn'] = bfn

    if dataset['binning_fn'] in _group_functions:
        dataset['binning_fn'] = _group_functions[dataset['binning_fn']]

    g_field = get_column_or_default(dataset['data'], 'binning_field')
    df = gcg.hexify_geodataframe(dataset['data'], hex_resolution=hex_resolution)
    logger.debug(f'hexagonal overlay added, res={hex_resolution}.')
    dataset['hex_resolution'] = hex_resolution

    if g_field is None:
        df = generate_dataframe_random_ids(df)
        df.rename({'r-ids': 'binning_field'}, axis=1, inplace=True)

    try:
        if dataset['v_type'] == 'qualitative':
            df['value_field'] = df['value_field'].astype(str)
            dataset['v_type'] = 'str'
    except KeyError:
        dataset['v_type'] = get_column_type(df, 'binning_field')

    if not dataset['binning_fn']:
        dataset['binning_fn'] = _group_functions['bestworst'] if dataset['v_type'] == 'str' else \
            _group_functions['count']

    df = gcg.bin_by_hex(df, dataset['binning_fn'], *binning_args, binning_field='binning_field', add_geoms=True,
                        **binning_kw)
    df = gcg.conform_geogeometry(df, fix_polys=True)

    logger.debug(f'data points binned by hexagon, length={len(df)}.')

    df.rename({'value': 'value_field'}, axis=1, inplace=True)
    logger.debug('dataframe geometry conformed to GeoJSON standard.')

    dataset['v_type'] = get_column_type(df, 'value_field')

    logger.debug(f'recognized data type, type={dataset["v_type"]}')
    dataset['data'] = df


def _prepare_general_dataset(name: str, dataset: DataSet, override_text=True):
    try:
        dataset['data'] = butil.get_shapes_from_world(dataset['data'])
    except (KeyError, ValueError, TypeError):
        logger.debug("If name was a country or continent, the process failed.")

    _read_dataset(dataset)
    dataset['data'] = gcg.conform_geogeometry(dataset['data'], fix_polys=dataset.get('validate', True))[
        ['geometry']]
    logger.debug('dataframe geometry conformed to GeoJSON standard.')

    if dataset.get('to_boundary', False):
        geom = dataset['data'].unary_union.boundary
        polys = [Polygon(list(g.coords)) for g in geom]
        mpl = MultiPolygon(polys).boundary
        dataset['data'] = GeoDataFrame({'geometry': [mpl]}, crs='EPSG:4326')
    if override_text:
        dataset['data']['text_field'] = name
    dataset['data']['value_field'] = 0


def _reset_adata(dataset: DataSet):
    dataset['adata'] = dataset['data'].copy(deep=True)


class PlotBuilder:
    """This class contains a Builder implementation for visualizing Plotly Hex data.
    """

    # default choropleth manager (includes properties for choropleth only)
    _default_dataset_manager: ClassVar[DataSetManager] = dict(
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
    _default_point_manager: ClassVar[DataSetManager] = dict(
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

    _default_grid_manager: ClassVar[DataSetManager] = dict(
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

    _default_region_manager: ClassVar[DataSetManager] = dict(
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

    _default_outline_manager: ClassVar[DataSetManager] = dict(
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
    _default_figure_manager: ClassVar[FigureManager] = dict(
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

    _default_file_output_manager: ClassVar[FileOutputManager] = dict(
        format='pdf', scale=1, height=500, width=600, engine='kaleido', validate=False
    )

    _default_regions: ClassVar[Dict[str, Dict]] = dict(canada=dict(
        data='CANADA'
    ))

    _default_grids: ClassVar[Dict[str, Dict]] = dict(canada=dict(
        data='CANADA'
    ))
    # _default_grids = {}

    _default_hex_resolution: ClassVar[int] = 3
    _default_range_buffer_lat: ClassVar[Tuple[float, float]] = (0.0, 0.0)
    _default_range_buffer_lon: ClassVar[Tuple[float, float]] = (0.0, 0.0)

    _default_plot_settings: ClassVar[Dict[str, Any]] = {
        'focus_mode': FocusMode.AUTO_BUILDER,
        'clip_mode': None,
        'clip_points': False,
        'auto_grid': False,
        'scale_mode': 'linear',
        'plot_regions': True,
        'plot_grids': True,
        'plot_points': True,
        'plot_outlines': True,
        'remove_empties': True,
        'replot_empties': True,
        'range_buffer_lat': (0.0, 0.0),
        'range_buffer_lon': (0.0, 0.0),
        'file_output': None,
        'show': True
    }

    @property
    def scale_mode(self):
        return self._plot_settings['scale_mode']

    @scale_mode.setter
    def scale_mode(self, scale_mode):
        self.set_scale_mode(scale_mode=scale_mode)

    def set_scale_mode(self, scale_mode=None, **kwargs):
        prevmode = self._plot_settings['scale_mode']
        scale_mode = scale_mode if scale_mode is not None else self._plot_settings['scale_mode']
        dataset = self._get_main_dataset()['data']
        if prevmode == 'logarithmic' and scale_mode == 'linear':
            butil.delogify_scale(dataset['data'])
            dict_deep_update(dataset['manager'], {'colorbar':
                {
                    'tickvals': None,
                    'ticktext': None
                },
                'zmin': None,
                'zmax': None
            })
        elif prevmode == 'linear' and scale_mode == 'logarithmic':
            dict_deep_update(dataset['manager'], butil.logify_scale(dataset['data'], **kwargs))
        elif prevmode == scale_mode:
            pass
        elif scale_mode not in ['linear', 'logarithmic']:
            raise ValueError(f"The scale mode you selected is invalid, mode={scale_mode}.")
        else:
            raise NotImplementedError(
                f"The scale can not be converted from {prevmode} to {scale_mode}, it has not been implemented yet.")

        logger.debug(f"The scale has been converted from {prevmode} to {scale_mode}.")

    def _recombine(self):
        if assume_changes:
            self._combine_regions()
            self._combine_grids()
            self._combine_points()
            self._combine_points()

    @property
    def clip_mode(self):
        return self._plot_settings['clip_mode']

    @clip_mode.setter
    def clip_mode(self, clip_mode):
        self.set_clip_mode(clip_mode=clip_mode)

    def _clip_dataset(self, dataset, operation):
        clip_mode = self._plot_settings['clip_mode']
        clip_args = clip_mode.split('+')

        alterations = []
        _reset_adata(dataset)
        if 'outlines' in clip_args:
            alterations.append(gcg.clip(dataset, self._comout, operation=operation))
        if 'grids' in clip_args:
            alterations.append(gcg.clip(dataset, self._comgrids, operation=operation))
        if 'regions' in clip_args:
            alterations.append(gcg.clip(dataset, self._comregs, operation=operation))

        return gcg.merge_datasets_simple(*alterations)

    def clip_datasets(self, clip_mode: str):
        clip_args = clip_mode.split('+')

        self._reset_adatas()
        self._recombine()

        dsclip = self._main_dataset
        alterations = {'dsalter': [], 'gsalter': defaultdict(list), 'psalter': defaultdict(list)}
        if 'outlines' in clip_args:

            alterations['dsalter'].append(gcg.clip_hexes_to_polygons(dsclip['adata'], self._comout))

            for k, v in self._grids.items():
                alterations['gsalter'][k].append(gcg.clip_hexes_to_polygons(v['adata'], self._comout))


            for k, v in self._points.items():
                alterations['psalter'][k].append(gcg.clip_hexes_to_polygons(v['adata'], self._comout))

            # alterations['gsalter'].append(
            #   gcg.clip_hexes_to_polygons(self._comgrids, self._comout))
            # alterations['psalter'].append(
            #   gcg.clip_points_to_polygons(self._compoi, self._comout))
        if 'grids' in clip_args:
            alterations['dsalter'].append(gcg.clip_hexes_to_hexes(dsclip['adata'], self._comgrids))
            for k, v in self._points.items():
                alterations['psalter'][k].append(gcg.clip_hexes_to_polygons(v['adata'], self._comgrids))

            # alterations['psalter'].append(
            #   gcg.clip_points_to_polygons(self._compoi, self._comgrids))
        if 'regions' in clip_args:
            alterations['dsalter'].append(gcg.clip_hexes_to_polygons(dsclip['adata'], self._comregs))

            for k, v in self._grids.items():
                print(v['adata'])
                print(gcg.clip(v['adata'], self._comregs))
                alterations['gsalter'][k].append(gcg.clip_hexes_to_polygons(v['adata'], self._comregs))
                print('GRALTER', alterations['gsalter'][k])

            for k, v in self._points.items():
                alterations['psalter'][k].append(gcg.clip_hexes_to_polygons(v['adata'], self._comregs))

            # alterations['gsalter'].append(
            #   gcg.clip_hexes_to_polygons(self._comgrids, self._comregs))
            # alterations['psalter'].append(
            #   gcg.clip_points_to_polygons(self._compoi, self._comregs))

        dsalter = alterations['dsalter']
        self._main_dataset['adata'] = gcg.repeater_merge(*dsalter, how='outer', on=['hex', 'value_field', 'geometry'])
        print(self._main_dataset['adata'])

        psalter = alterations['psalter']
        for k, v in self._points.items():
            v['adata'] = gcg.repeater_merge(*psalter[k], how='outer')

        gsalter = alterations['gsalter']
        for k, v in self._grids.items():
            v['adata'] = gcg.repeater_merge(*gsalter[k], how='outer', on=['hex', 'value_field', 'geometry'])
            print('GSS',v['adata'])
        print(gsalter)
        # self._recombine()

    def _reset_adatas(self):
        _reset_adata(self._main_dataset)
        for _, v in self._regions.items():
            _reset_adata(v)
        for _, v in self._grids.items():
            _reset_adata(v)
        for _, v in self._outlines.items():
            _reset_adata(v)
        for _, v in self._points.items():
            _reset_adata(v)

    @property
    def remove_empties(self):
        return self._plot_settings['remove_empties']

    @remove_empties.setter
    def remove_empties(self, remove_empties):
        self.set_remove_empties(remove_empties=remove_empties)

    def set_remove_empties(self, remove_empties=False):
        ds = self._main_dataset
        df = self._main_dataset['adata']
        if remove_empties:
            empty_symbol = 'empty' if ds['v_type'] == 'str' else 0
            empties = df[df['value_field'] == empty_symbol]
            df = df[df['value_field'] != empty_symbol]
            ds['adata'] = df
            logger.debug(f'removed empty rows from main dataset, length={len(df)}.')
            if self._plot_settings['replot_empties']:
                self._grids['*MAINEMPTIES*'] = {'data': empties, 'adata': empties.copy(deep=True)}
        else:
            ds['adata'] = pd.concat([ds['adata'], ds.pop('empties')])
            if not self._plot_settings['replot_empties']:
                try:
                    self._grids.pop('*MAINEMPTIES*')
                except KeyError:
                    logger.debug("Replot empties was set to false, but no previous plotted empties were found.")
        self._plot_settings['remove_empties'] = remove_empties

    @property
    def auto_grid(self):
        return self._plot_settings['remove_empties']

    @auto_grid.setter
    def auto_grid(self, auto_grid):
        self.set_auto_grid(auto_grid=auto_grid)

    def set_auto_grid(self, auto_grid=False):
        if auto_grid:
            self._make_auto_grid()
        else:
            try:
                self._grids.pop('*AUTOGRID*')
            except KeyError:
                logger.debug("Auto grid was set to false, but no previous auto grid was found.")

        self._plot_settings['auto_grid'] = auto_grid

    def _make_auto_grid(self):
        clip_mode = self._plot_settings['clip_mode']
        logger.debug('attempting to get auto grid dataset.')

        self._recombine()

        if clip_mode == 'regions':
            grid = gcg.hexify_geodataframe(self._comregs)
        elif clip_mode == 'outlines':
            grid = gcg.hexify_geodataframe(self._comout)
        elif isvalid_dataset(self._main_dataset):
            grid = gcg.generate_grid_over_hexes(self._main_dataset['adata'])
        else:
            raise gce.BuilderDatasetInfoError("There is no dataset to form an auto-grid with!")

        grid = grid.rename({'value': 'value_field'}, axis=1, errors='raise')
        self._grids['*AUTOGRID*'] = {'data': grid, 'adata': grid.copy(deep=True)}

    def __init__(self, plot_name: Optional[str] = None, main_dataset: Optional[DataSet] = None,
                 grids: Optional[DataSets] = None, regions: Optional[DataSets] = None,
                 outlines: Optional[DataSets] = None,
                 points: Optional[DataSets] = None, default_grids: Tuple[bool, int] = (True, 3),
                 default_regions: bool = True, logging_file: Optional[str] = None):
        """Initializer for instances of GeoCanVisualize.

        Initializes the new instance with the given hex objects,
        which are dicts. Sets the builder's settings to defaults,
        which can be changed later.

        :param plot_name: The name of the plot for builder purposes
        :type plot_name: Optional[str]
        :param main_dataset: The datasets to be plotted
        :type main_dataset: Optional[DataSet]
        :param grids: The datasets to add to the builder (will not turned into empty grids)
        :type grids: Optional[DataSets]
        :param points: The datasets to add to the builder (will not turned into scatter plots)
        :type points: Optional[DataSets]
        :param regions: The datasets to add to the builder (regions that will be highlighted)
        :type regions: Optional[DataSets]
        """

        self._output_stats = {}
        self.plot_name = plot_name

        if logging_file:
            loggingAddFile(fix_filepath(logging_file, add_filename=self.plot_name or 'temp', add_ext='log'))
            toggleLogging(True)
            logger.debug('logger set.')

        self._figure: Figure = Figure()
        logger.debug('initialized internal figure.')

        """
        We need an efficient way to alter the data.
        1) Group the data together in a single dataframe and alter all at once
        2) Alter each dataset individually
        """

        self._output_stats = {}
        self._clear_figure()
        self._dataset_manager = {}
        self._grid_manager = {}
        self._figure_manager = {}
        self._file_output_manager = deepcopy(self._default_file_output_manager)
        self.hex_resolution = self._default_hex_resolution
        self.range_buffer_lat = (0.0, 0.0)
        self.range_buffer_lon = (0.0, 0.0)
        self._hone_geometry = []
        self._plot_settings = deepcopy(self._default_plot_settings)

        self._main_dataset = None
        self._grids: DataSets = {}
        self._points: DataSets = {}
        self._regions: DataSets = {}
        self._outlines: DataSets = {}

        self._comregs = GeoDataFrame()
        self._comgrids = GeoDataFrame({'hex': [], 'geometry': []})
        self._comout = GeoDataFrame()
        self._compoi = GeoDataFrame()

        if main_dataset is not None:
            self.main_dataset = main_dataset

        if regions:
            for k, v in regions.items():
                self.add_region(k, v)

        if grids:
            for k, v in grids.items():
                self.add_grid(k, v)

        if outlines:
            for k, v in outlines.items():
                self.add_outline(k, v)

        if points:
            for k, v in points.items():
                self.add_point(k, v)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def update_plot_settings(self, settings: Optional[Dict[str, Any]] = None, **kwargs):
        """Updates the plot settings of the builder.

        :param settings: The settings to update
        :type settings: Optional[Dict[str, Any]]
        :param kwargs: Any additional settings to update
        :type kwargs: **kwargs
        """
        if settings:
            self._plot_settings.update(settings)
        self._plot_settings.update(kwargs)
        for item in self._default_plot_settings.items():
            if item[0] not in self._plot_settings:
                self._plot_settings[item[0]] = item[1]
        if settings or len(kwargs) > 0:
            logger.debug('plot settings updated.')

    @property
    def main_dataset(self):
        return deepcopy(self._main_dataset)

    def _get_main_dataset(self):
        return self._main_dataset

    """
    Perhaps data sets could be clipped upon entry.
    
    Override existing data when plot settings are updated (or not)
    """

    def set_main_dataset(self, dataset: DataSet):
        """Sets the main dataset of the builder.

        bins are dicts formatted as such:
        <name of bin> = {
            frame = <filename | GeoDataFrame | DataFrame>
            latitude_field = <lat column in dataframe> (Opt)
            longitude_field = <lon column in dataframe> (Opt)
            geometry_field = <geometry column in dataframe> (Opt)
            hex_resolution = <hex resolution for the dataframe (0-15)> (Opt)
            binning_field = <column to group the data by> (Opt)
            binning_fn = <the function to apply to the grouped data (lambda)> (Opt)
            manager = <dict of management properties for the plotting of this bin> (Opt)
        }

        Keep in mind only certain things in the above example should be present when
        the user inputs their dict.
        """
        logger.debug('began conversion of main dataset.')
        _read_dataset(dataset)
        _hexbinify_dataset(dataset, self.hex_resolution)
        logger.debug('ended conversion of main dataset.')
        dataset['adata'] = dataset['data'].copy(deep=True)
        dataset['empties'] = GeoDataFrame()
        self._main_dataset = dataset

    @main_dataset.setter
    def main_dataset(self, dataset: DataSet):
        self.set_main_dataset(dataset)

    def add_grid(self, name: str, dataset: DataSet):
        """Adds a grid data set to the builder.

        The minimum attributes for a grid data set are as follows:
        1) if data is a filepath
        {
            'data': <filepath>

            AND

            'latitude_field': <column containing latitudes>
            'longitude_field': <column containing longitudes>

            OR (if geometry isn't loaded)

            'geometry_field': <column containing geometries>
        }

        2) if data is a dataframe (same as 1)
        3) if data is a geodataframe (same as 1, without latitude_field, longitude_field, and geometry_field)


        :param name: The name this dataset will be referred to as
        :type name: str
        :param dataset: The grid dataset to add
        :type dataset: DataSet
        """

        logger.debug(f'began conversion of grid, name={name}.')
        _prepare_general_dataset(name, dataset)
        dataset['binning_field'] = 'value_field'
        _hexbinify_dataset(dataset, self.hex_resolution)

        dataset['data']['value_field'] = 0
        dataset['adata'] = dataset['data'].copy(deep=True)
        dataset['data']['*DS_NAME*'] = name
        self._comgrids = gcg.repeater_merge(self._comgrids, dataset['adata'], on=['hex', 'geometry'])
        dataset['data'].drop(columns='*DS_NAME*', inplace=True)

        self._grids[name] = dataset
        logger.debug(f'ended conversion of grid, name={name}.')

    def add_outline(self, name: str, dataset: DataSet):
        """Adds a outline data set to the builder.

        The minimum attributes for a outline data set are as follows:
        1) if data is a filepath
        {
            'data': <filepath>

            AND

            'latitude_field': <column containing latitudes>
            'longitude_field': <column containing longitudes>

            OR (if geometry isn't loaded)

            'geometry_field': <column containing geometries>
        }

        2) if data is a dataframe (same as 1)
        3) if data is a geodataframe (same as 1, without latitude_field, longitude_field, and geometry_field)

        :param name: The name this dataset will be referred to as
        :type name: str
        :param dataset: The grid dataset to add
        :type dataset: DataSet
        """

        logger.debug(f'began conversion of outline, name={name}.')
        _prepare_general_dataset(name, dataset)
        dataset['data'] = gcg.pointify_geodataframe(dataset['data'], keep_geoms=False)
        dataset['adata'] = dataset['data'].copy(deep=True)
        dataset['data']['*DS_NAME*'] = name
        self._comout = pd.concat([self._comout, dataset['data']])
        dataset['data'].drop(columns='*DS_NAME*', inplace=True)
        self._outlines[name] = dataset

        logger.debug(f'ended conversion of outline, name={name}.')

    def add_region(self, name: str, dataset: DataSet):
        """Adds a region data set to the builder.

        The minimum attributes for a region data set are as follows:
        1) if data is a shapefile filepath
        {
            'data': <filepath>
        }

        :param name: The name this dataset will be referred to as
        :type name: str
        :param dataset: The grid dataset to add
        :type dataset: DataSet
        """
        logger.debug(f'began conversion of region, name={name}.')
        _prepare_general_dataset(name, dataset)
        dataset['adata'] = dataset['data'].copy(deep=True)
        dataset['data']['*DS_NAME*'] = name
        self._comregs = pd.concat([self._comregs, dataset['data']])
        dataset['data'].drop(columns='*DS_NAME*', inplace=True)
        self._regions[name] = dataset
        logger.debug(f'ended conversion of region, name={name}.')

    def add_region_from_world(self, name, store_as: Optional[str] = None):
        region_dataset = {'data': butil.get_shapes_from_world(name)}
        store_as = store_as if store_as else name
        self.add_region(store_as, region_dataset)

    def add_point(self, name: str, dataset: DataSet):
        """Adds a point hex to the builder.

        :param name: The name this dataset will be referred to as
        :type name: str
        :param dataset: The point dataset to add
        :type dataset: DataSet
        """
        logger.debug(f'began conversion of points, name={name}.')
        _prepare_general_dataset(name, dataset, override_text=False)

        dataset['data'] = gcg.pointify_geodataframe(dataset['data'], keep_geoms=False)
        dataset['adata'] = dataset['data'].copy(deep=True)
        dataset['data']['*DS_NAME*'] = name
        self._compoi = pd.concat([self._compoi, dataset['data']])
        dataset['data'].drop(columns='*DS_NAME*', inplace=True)
        self._points[name] = dataset
        logger.debug(f'ended conversion of points, name={name}.')

    def _combine_points(self):
        self._compoi = make_multi_dataset(self._points)

    def _combine_regions(self):
        self._comregs = make_multi_dataset(self._regions)

    def _combine_grids(self):
        try:
            self._comgrids = gcg.repeater_merge(*[ds['adata'] for ds in list(self._grids.values())], on=['hex', 'geometry'])
            #self._comgrids = gcg.merge_datasets_simple(*[ds['adata'] for ds in list(self._grids.values())],
                                                       #result_name='value_field')
        except IndexError:
            pass

    def _combine_outlines(self):
        self._comout = make_multi_dataset(self._outlines)

    def update_dataset_manager(self, manager: DataSetManager):
        """Updates the manager of the bin dataset with the given name.

        If the name given is None, then the general hex manager is updated,
        and every grid currently in the builder will be set to the same
        manager.

        :param manager: The updates for this dataset's manager
        :type manager: DataSetManager
        """
        dict_deep_update(self._dataset_manager, manager)
        logger.debug('main dataset manager updated.')

    def update_grid_manager(self, manager: DataSetManager):
        """Updates the manager of the grid dataset with the given name.

        If the name given is None, then the general hex manager is updated,
        and every grid currently in the builder will be set to the same
        manager.

        :param manager: The updates for this dataset's manager
        :type manager: DataSetManager
        """
        dict_deep_update(self._grid_manager, manager)
        logger.debug('grid dataset manager updated.')

    def update_point_manager(self, manager: DataSetManager, name: Optional[str] = None):
        """Updates the manager of the region dataset with the given name.

        If the name given is None, then the general region manager is updated,
        and every region currently in the builder will be set to the same
        manager.

        :param manager: The updates for this dataset's manager
        :type manager: DataSetManager
        :param name: The name of the region to updat
        :type name:
        """

        if name:
            dict_deep_update((self._points[name])['manager'], manager)
        else:
            for pname in self._points:
                dict_deep_update(self._points[pname]['manager'], manager)
        logger.debug(f'point dataset manager updated, name={name}.')

    def update_region_manager(self, manager: DataSetManager, name: Optional[str] = None):
        """Updates the manager of the region dataset with the given name.

        If the name given is None, then the general region manager is updated,
        and every region currently in the builder will be set to the same
        manager.

        :param manager: The updates for this dataset's manager
        :type manager: DataSetManager
        :param name: The name of the region to updat
        :type name:
        """
        if name:
            dict_deep_update((self._regions[name])['manager'], manager)
        else:
            for rname in self._regions:
                dict_deep_update(self._regions[rname]['manager'], manager)
        logger.debug(f'region dataset manager updated, name={name}.')

    def update_outline_manager(self, manager: DataSetManager, name: Optional[str] = None):
        """Updates the manager of the region dataset with the given name.

        If the name given is None, then the general region manager is updated,
        and every region currently in the builder will be set to the same
        manager.

        :param manager: The updates for this dataset's manager
        :type manager: DataSetManager
        :param name: The name of the region to updat
        :type name:
        """

        if name:
            dict_deep_update((self._outlines[name])['manager'], manager)
        else:
            for oname in self._outlines:
                dict_deep_update(self._outlines[oname]['manager'], manager)
        logger.debug(f'outline dataset manager updated, name={name}.')

    def update_figure_manager(self, manager: FigureManager):
        """Updates the manager of the figure.

        The updates dict should be formatted as follows:
        {
            geos = <dict of Plotly geos properties> (Opt)
            layout = <dict of Plotly layout properties> (Opt)
            traces = <dict of Plotly traces properties> (Opt)
        }

        :param manager: The updates for the figure's manager
        :type manager: FigureManager
        """

        dict_deep_update(self._figure_manager, manager)
        logger.debug('figure manager updated.')

    def update_file_output_manager(self, manager: FileOutputManager):
        """Updates the manager for file output.

        The updates dict should be formatted as follows:
        {
            format = <format of output> (Opt)
            scale = <integer scale of output> (Opt)
            width = <width of file output (px)> (Opt)
            height = <height of output (px)> (Opt)
            validate = <validation of file output> (Opt)
        }

        :param manager: The updates for this region hex's manager
        :type manager: FileOutputManager
        """
        self._file_output_manager.update(manager)
        logger.debug('file output manager updated.')

    def _focus(self):
        """Zooms in on the area of interest according to the builder.

        Sets the lataxis,lonaxis,projection

        :param geometry_col: A list of geometry if auto-builder is being used
        :type geometry_col: Optional[List[Union[Point,Polygon]]]
        """
        fm = FocusMode(self._plot_settings['focus_mode'])
        if fm is FocusMode.AUTO_BUILDER:
            self._hone()
        elif fm is FocusMode.AUTO_FITBOUND:
            self._figure.update_geos(fitbounds='locations')
        logger.debug(f'focused internal figure, mode={fm}')

    def _hone(self):
        """Fits boundaries to the given geometry column.

        Performs mathematical computations to fit boundaries to
        the dataset.
        """

        cen = gcg.find_center_simple(self._hone_geometry)
        rng = gcg.find_ranges_simple(self._hone_geometry)

        buffer_lat = self._plot_settings['range_buffer_lat']
        buffer_lon = self._plot_settings['range_buffer_lon']

        cd_lat = None
        cd_lon = None
        projection_rotation = None
        center = None

        if rng is not None:
            lat_r = list(rng[1])
            lat_r[0] = lat_r[0] - buffer_lat[0]
            lat_r[1] = lat_r[1] + buffer_lat[1]
            cd_lat = lat_r

            lon_r = list(rng[0])
            lon_r[0] = lon_r[0] - buffer_lon[0]
            lon_r[1] = lon_r[1] + buffer_lon[1]
            cd_lon = lon_r

        if cen is not None:
            projection_rotation = dict(lon=cen.x, lat=cen.y)
            center = dict(lon=cen.x, lat=cen.y)

        projection = self._figure.layout.geo.projection.type
        if projection in ['orthographic', 'azimuthal equal area']:
            self._figure.update_geos(projection_rotation=projection_rotation,
                                     lataxis_range=cd_lat, lonaxis_range=cd_lon)
        else:
            self._figure.update_geos(center=center, projection_rotation=projection_rotation,
                                     lataxis_range=cd_lat, lonaxis_range=cd_lon)

    def output_to_file(self, filetype: str = 'png'):
        """Opens the html for the figure and saves a lower quality image.

        :param filetype: The filetype for the saved image (png, jpg, etc.)
        :type filetype: str
        """
        plotly.offline.plot(self._figure, output_type='file', image=filetype, auto_open=True)

    def _output_figure(self, show: bool = True, file_output: Optional[str] = None):
        """Displays or outputs the internal figure

        :param show: Whether to show the final figure or not
        :type show: bool
        :param file_output: A filename for output
        :type file_output: Optional[str]
        :return: The final figure (show=False)
        :rtype: Figure
        """

        output = None
        if show:
            self._figure.show(renderer='browser')
        else:
            output = self._figure

        if file_output:
            self._figure.write_image(fix_filepath(file_output, add_filename=self.plot_name,
                                                  add_ext=self._file_output_manager.get('format', '')),
                                     **self._file_output_manager)

        logger.debug('figure output successfully.')
        return output

    def print_datasets(self):
        """Prints the datasets currently within the builder.
        """
        print('[MAIN DATASET]\n', self._main_dataset, '\n')
        print('[GRIDS]\n', self._grids, '\n')
        print('[POINTS]\n', self._points, '\n')
        print('[REGIONS]\n', self._regions, '\n')

    def reset(self):
        """Resets the entire builder.
        """
        self._output_stats = {}
        self._clear_figure()
        self._dataset_manager = {}
        self._grid_manager = {}
        self._figure_manager = {}
        self._file_output_manager = deepcopy(self._default_file_output_manager)
        self.hex_resolution = self._default_hex_resolution
        self.range_buffer_lat = (0.0, 0.0)
        self.range_buffer_lon = (0.0, 0.0)
        self._hone_geometry = []
        self._plot_settings = deepcopy(self._default_plot_settings)

        self._datasets: DataSets = {}
        self._grids: DataSets = {}
        self._points: DataSets = {}
        self._regions: DataSets = {}

    def _clear_figure(self):
        """Clears the internal figure of the builder.
        """
        self._figure.data = []
        logger.debug('internal figure data cleared.')

    def remove_grid(self, name: str) -> DataSet:
        """Removes a grid dataset from the builder.

        :param name: The name of the grid
        :type name: str
        :return: The removed grid
        :rtype: dict
        """
        logger.debug(f'grid removed, name={name}')
        return self._grids.pop(name)

    def remove_point(self, name: str) -> DataSet:
        """Removes a point dataset from the builder.

        :param name: The name of the point
        :type name: str
        :return: The removed point
        :rtype: dict
        """
        logger.debug(f'points removed, name={name}')
        return self._points.pop(name)

    def remove_region(self, name: str) -> DataSet:
        """Removes a region hex from the builder.

        :param name: The name of the region
        :type name: str
        :return: The removed region
        :rtype: dict
        """
        logger.debug(f'region removed, name={name}')
        return self._regions.pop(name)

    def clear_dataset_manager(self):
        """Clears the main dataset's manager (empty manager)
        """
        self._dataset_manager.clear()
        logger.debug('main dataset manager cleared.')

    def clear_grid_manager(self):
        """Clears the general grid manager (empty manager)
        """
        self._grid_manager.clear()
        logger.debug('grid manager cleared.')

    def clear_point_manager(self, name: Optional[str] = None):
        """Clears the point manager (empty manager)

        If no name is given, the general point manager is cleared.

        :param name: The name of the dataset to clear the manager of
        :type name: Optional[str]
        """
        if name is not None:
            (self._points[name])['manager'].clear()
            logger.debug(f'points manager cleared, name={name}.')
        else:
            for poiname in self._points:
                self._regions[poiname]['manager'].clear()
                logger.debug(f'points manager cleared, name={poiname}.')

    def clear_region_manager(self, name: Optional[str] = None):
        """Clears the region manager (empty manager)

        If no name is given, the general region manager is cleared.

        :param name: The name of the dataset to clear the manager of
        :type name: Optional[str]
        """
        if name:
            (self._regions[name])['manager'].clear()
            logger.debug(f'region manager cleared, name={name}.')
        else:
            for regname in self._regions:
                self._regions[regname]['manager'].clear()
                logger.debug(f'region manager cleared, name={regname}.')

    def clear_outline_manager(self, name: Optional[str] = None):
        """Clears the outline manager (empty manager)

        If no name is given, the general region manager is cleared.

        :param name: The name of the dataset to clear the manager of
        :type name: Optional[str]
        """
        if name is not None:
            (self._outlines[name])['manager'].clear()
            logger.debug(f'outline manager cleared, name={name}.')
        else:
            for outname in self._outlines:
                self._outlines[outname]['manager'].clear()
                logger.debug(f'outline manager cleared, name={outname}.')

    def clear_figure_manager(self):
        """Clears the figure manager (empty manager)
        """
        self._figure_manager = dict(geos=dict(),
                                    layout=dict(),
                                    traces=dict())
        logger.debug('figure manager cleared.')

    def clear_file_output_manager(self):
        """Clears the region manager (empty manager)
        """
        self._file_output_manager.clear()
        logger.debug('file output cleared.')

    def _make_general_scatter_trace(self, gdf: GeoDataFrame, initial_properties: DataSetManager,
                                    final_properties: Optional[DataSetManager] = None,
                                    disjoint: bool = True) -> Scattergeo:
        """Adds a generic scatter trace to the plot.

        :param gdf: The GeoDataFrame containing the geometries to plot
        :type gdf: GeoDataFrame
        :param initial_properties: The initial set of plotly properties to be applied to the Scattergeo object
        :type initial_properties: DataSetManager
        :param final_properties: The set of plotly properties to apply after
        :type final_properties: Optional[DataSetManager]
        :param disjoint: Whether the latitudes and longitudes should be disjoint by Nan entries or not
        :type disjoint: bool
        :return: The trace to be added to the plot
        :rtype: Scattergeo
        """

        lats, lons = butil.to_plotly_points_format(gdf, disjoint=disjoint)
        scatt = Scattergeo(
            lat=lats,
            lon=lons
        ).update(initial_properties).update(final_properties if final_properties else {})

        logger.debug('scattergeo trace generated.')
        return scatt

    def _make_general_trace(self, gdf: GeoDataFrame, initial_properties: DataSetManager,
                            final_properties: Optional[DataSetManager] = None):
        """Adds a generic choropleth trace to the internal figure.

        :param gdf: The GeoDataFrame to plot
        :type gdf: GeoDataFrame
        """

        geojson = gcg.simple_geojson(gdf, 'value_field')

        choro = Choropleth(
            locations=gdf.index,
            z=gdf['value_field'],
            geojson=geojson,
            legendgroup='choros'
        ).update(initial_properties).update(final_properties if final_properties else {})

        logger.debug('choropleth trace generated.')
        return choro

    def _plot_dataset_traces(self):
        """Plots the trace of the main dataset onto the internal figure.

        :param ds: The main dataset (after alteration)
        :type ds: DataSet
        """

        dgeoms = []
        try:
            ds = self._main_dataset

            self._output_stats['main-dataset'] = defaultdict(dict)
            logger.debug('adding main dataset to plot.')
            df = ds['adata']
            print(len(df))

            try:
                bounds = min(df['value_field']), max(df['value_field'])
            except TypeError:
                pass

            try:
                cs = self._dataset_manager.pop('colorscale')
            except KeyError:
                cs = self._default_dataset_manager['colorscale']

            try:
                cs['scale_value'] = getScale(cs['scale_value'], cs['scale_type'])
            except (AttributeError, KeyError):
                try:
                    cs['scale_value'] = tryGetScale(cs['scale_value'])
                except AttributeError:
                    pass
            except TypeError:
                try:
                    cs = {'scale_value': tryGetScale(cs)}
                except AttributeError:
                    cs = {'scale_value': cs}

            plot_dfs = []

            if ds['v_type'] == 'num':

                df_prop = deepcopy(self._dataset_manager)
                print(df)
                logger.debug(
                        f'quantitative dataset trace added, (min,max)={str(tuple([min(df["value_field"]), max(df["value_field"])]))}.')
                df_prop['text'] = df['text'] = 'VALUE: ' + df['value_field'].astype(str)

                discrete = self._plot_settings.pop('discrete_scale', cs.pop('discrete', False))
                if discrete:
                    if isinstance(discrete, dict):
                        cs['scale_type'] = cs.get('scale_type', 'sequential')
                        cs['scale_value'] = getDiscreteScale(cs['scale_value'], cs['scale_type'],
                                                             *bounds,
                                                             **discrete)
                    elif isinstance(discrete, bool):
                        cs['scale_value'] = getDiscreteScale(cs['scale_value'], cs['scale_type'],
                                                             *bounds)
                    else:
                        raise gce.BuilderDatasetInfoError("Discrete should be a boolean or dict-type argument.")
                else:
                    df_prop['zmin'], df_prop['zmax'] = bounds

                df_prop['colorscale'] = cs['scale_value']
                plot_dfs.append(('quantitative', df, df_prop))

            elif ds['v_type'] == 'str':

                df['value_field'] = df['value_field'].fillna('empty').astype(str)
                df['temp-count'] = 0
                ds['data'] = df

                cs['scale_value'] = solidScales(cs['scale_value'])
                for i, name in enumerate(sorted(df['value_field'].unique())):
                    dfp_prop = deepcopy(self._dataset_manager)
                    dfp_prop.update(dict(
                        name=name, colorscale=cs['scale_value'][i], text=f'BEST OPTION: {name}',
                        showlegend=True, showscale=False))
                    plot_dfs.append(('qualitative', df[df['value_field'] == name].drop(columns='value_field')
                                     .rename({'temp-count': 'value_field'}, axis=1), dfp_prop))

            conform_alpha = self._plot_settings.pop('conform_alpha', cs.pop('conform_alpha', True))
            self._plot_settings.pop('discrete_scale', cs.get('discrete', False))

            for i, (typer, plot_df, prop) in enumerate(plot_dfs):

                self._output_stats['main-dataset'][typer].update({i: get_stats(plot_df, hexed=True)}, )

                choro = self._make_general_trace(plot_df, self._default_dataset_manager, final_properties=prop)

                if conform_alpha:
                    choro.colorscale = configureScaleWithAlpha(list(choro.colorscale), float(choro.marker.opacity))

                self._figure.add_trace(choro)
                logger.debug(f'{typer} dataset trace added.')

            self._output_stats['main-dataset'] = dict(self._output_stats['main-dataset'])
            dgeoms = list(df.geometry)
        except (KeyError, ValueError):
            pass
        return dgeoms

    def _remove_underlying_grid(self, df: GeoDataFrame, gdf: GeoDataFrame):
        """Removes the grid from underneath the data (lower file size).

        :param ds: The main dataset (after altreration)
        :type ds: DataSet
        :param gds: The merged grid dataset (after alteration)
        :type gds: DataSet
        """

        if isvalid_dataframe(df) and isvalid_dataframe(gdf):
            gdf = gpd.overlay(gdf, df[['value_field', 'geometry']],
                              how='difference')
            gdf = gcg.remove_other_geometries(gdf, 'Polygon')
            return gdf

    def _plot_regions(self):
        rgeoms = []

        if self._plot_settings['plot_regions']:
            logger.debug('adding regions to plot.')
            try:
                self._output_stats['regions'] = {}
                initial_prop = deepcopy(self._default_region_manager)
                for regname, regds in self._regions.items():
                    print('ADDING REG', regname, regds)
                    self._output_stats['regions'][regname] = get_stats(regds['adata'])
                    regds['manager']['text'] = regname
                    initial_prop['text'] = regname
                    self._figure.add_trace(
                        self._make_general_trace(regds['adata'], initial_prop, final_properties=regds['manager']))
                    rgeoms.extend(list(regds['adata'].geometry))
            except KeyError:
                print('KERR')
                pass
        return rgeoms

    def _plot_outlines(self):

        ogeoms = []
        if self._plot_settings['plot_outlines']:
            print('adding outlines to plot')
            logger.debug('adding outlines to plot.')
            try:
                self._output_stats['outlines'] = {}
                initial_prop = deepcopy(self._default_outline_manager)
                for outname, outds in self._outlines.items():
                    self._output_stats['outlines'][outname] = get_stats(outds['data'])
                    initial_prop['name'] = outname
                    initial_prop['text'] = f'{outname}-outline'

                    self._figure.add_trace(
                        self._make_general_scatter_trace(outds['adata'], initial_prop,
                                                         final_properties=outds['manager'], disjoint=True))
                    ogeoms.extend(list(outds['adata'].geometry))
            except KeyError:
                pass
        return ogeoms

    def _plot_grids(self):

        ggeoms = []
        if self._plot_settings['plot_grids']:
            logger.debug('adding grids to plot.')
            try:
                gds = self._comgrids
                logger.debug('adding grids to plot.')
                gds = self._remove_underlying_grid(self._main_dataset['adata'], gds)

                gds['text_field'] = 'GRID'
                # self._output_stats['grids'] = get_stats(gds, hexed=True)
                initial_prop = deepcopy(self._default_grid_manager)
                initial_prop['text'] = 'GRID'
                self._figure.add_trace(
                    x := self._make_general_trace(gds, initial_prop, final_properties=self._grid_manager))
                ggeoms = list(gds.geometry)
            except KeyError as e:
                raise e
                pass
        return ggeoms

    def _plot_points(self):
        pgeoms = []
        if self._plot_settings['plot_grids']:
            logger.debug('adding points to plot.')
            try:
                self._output_stats['points'] = {}
                initial_prop = deepcopy(self._default_point_manager)
                for poiname, poids in self._points.items():
                    self._output_stats['points'][poiname] = get_stats(poids['adata'])
                    initial_prop['name'] = poiname
                    # initial_prop['hovertext'] = _get_hover_field(poids['adata'], poids['hover_fields'])
                    initial_prop['text'] = get_column_or_default(poids['adata'], 'text_field')
                    self._figure.add_trace(
                        self._make_general_scatter_trace(poids['adata'], initial_prop,
                                                         final_properties=poids['manager'],
                                                         disjoint=False))
                    pgeoms.extend(list(poids['adata'].geometry))
            except KeyError:
                pass
        return pgeoms

    def _plot_traces(self):
        """Plots each of the merged dataset collections.

        This must be performed

        :param ds: The main dataset after alteration
        :type ds: DataSet
        :param rds: The merged regions dataset
        :type rds: DataSet
        :param gds: The merged grids dataset
        :type gds: DataSet
        :param pds: The merged points dataset
        :type pds: DataSet
        """

        rgeoms = self._plot_regions()
        ggeoms = self._plot_grids()
        dgeoms = self._plot_dataset_traces()
        ogeoms = self._plot_outlines()
        pgeoms = self._plot_points()

        if not dgeoms:
            self._hone_geometry.extend(ogeoms if ogeoms else ggeoms if ggeoms else rgeoms)
        else:
            self._hone_geometry.extend(dgeoms)

        '''
                print('HERE', self._points)

        rgeoms = []
        if regions_on and isvalid_dataset(rds):
            logger.debug('adding regions to plot.')
            self._output_stats['regions'] = {}
            newregs = dissolve_multi_dataset(rds, self._regions)
            initial_prop = deepcopy(self._default_region_manager)
            for regname, regds in newregs.items():
                self._output_stats['regions'][regname] = get_stats(regds['data'])
                regds['manager']['text'] = regname
                initial_prop['text'] = regname
                self._figure.add_trace(
                    self._make_general_trace(regds['data'], initial_prop, final_properties=regds['manager']))

            rgeoms = list(rds['data'].geometry)

        ogeoms = []
        if outlines_on and isvalid_dataset(ods):
            logger.debug('adding outlines to plot.')
            self._output_stats['outlines'] = {}
            initial_prop = deepcopy(self._default_outline_manager)
            for outname, outds in dissolve_multi_dataset(ods, self._outlines).items():
                self._output_stats['outlines'][outname] = get_stats(outds['data'])
                initial_prop['name'] = outname
                initial_prop['text'] = f'{outname}-outline'
                self._figure.add_trace(
                    self._make_general_scatter_trace(outds['data'], initial_prop, final_properties=outds['manager']))

        ggeoms = []
        if grids_on and isvalid_dataset(gds):
            logger.debug('adding grids to plot.')
            self._remove_underlying_grid(ds, gds)

            merged_grids = (gds['data'])
            merged_grids['text_field'] = 'GRID'
            self._output_stats['grids'] = get_stats(gds['data'], hexed=True)
            initial_prop = deepcopy(self._default_grid_manager)
            initial_prop['text'] = 'GRID'
            self._figure.add_trace(
                self._make_general_trace(merged_grids, initial_prop, final_properties=self._grid_manager))
            ggeoms = list(merged_grids.geometry)

        dgeoms = []
        try:
            dgeoms = self._plot_dataset_traces(ds)
        except ValueError:
            pass

        if not dgeoms:
            self._hone_geometry.extend(ggeoms if ggeoms else rgeoms)
        else:
            self._hone_geometry.extend(dgeoms)

        if points_on and isvalid_dataset(pds):
            logger.debug('adding points to plot.')
            self._output_stats['points'] = {}
            initial_prop = deepcopy(self._default_point_manager)
            for poiname, poids in dissolve_multi_dataset(pds, self._points).items():
                print('TFTFTF', poiname, poids)
                self._output_stats['points'][poiname] = get_stats(poids['data'])
                initial_prop['name'] = poiname
                initial_prop['hovertext'] = _get_hover_field(poids['data'], poids['hover_fields'])
                initial_prop['text'] = get_column_or_default(poids['data'], 'text_field')
                self._figure.add_trace(
                    self._make_general_scatter_trace(poids['data'], initial_prop, final_properties=poids['manager'],
                                                     disjoint=False))
        '''

    def _make_multi_region_dataset(self):

        if self._plot_settings['plot_regions'] or self._plot_settings['clip_mode'] == 'regions':
            try:
                self._regions['*COMBINED*'] = make_multi_dataset(self._regions)
            except ValueError:
                self._regions['*COMBINED*'] = GeoDataFrame()

    def _make_multi_outline_dataset(self):

        if self._plot_settings['plot_outlines'] or self._plot_settings['clip_mode'] == 'outlines':
            self._outlines['*COMBINED*'] = make_multi_dataset(self._outlines)

    def _make_multi_point_dataset(self):

        if self._plot_settings['plot_points'] or self._plot_settings['clip_mode'] == 'points':
            self._points['*COMBINED*'] = make_multi_dataset(self._points)

    # TODO: tomorrow make it so the traces can represent multiple different regions
    def build_plot(self, **kwargs):
        """Builds the plot by adding the component traces.

        :param kwargs: Plot settings
        :type kwargs: **kwargs
        """

        # TODO: make combined dataframes contain empty geodataframes, Plotly doesn't care if it is given empty data

        '''
        try:
            rds = self._get_region_dataset()
        except gce.BuilderPlotBuildError as e:
            logger.warning(f'{e.message} Ignoring and returning figure...')
            rds = None
        '''
        self._plot_regions()
        self._plot_dataset_traces()
        # self._plot_grids()
        self._plot_outlines()
        self._plot_points()

        update_figure_plotly(self._figure, self._default_figure_manager)
        update_figure_plotly(self._figure, self._figure_manager)
        # self._focus()

        logger.info(f'Final Output Stats: {self._output_stats}')
        return self._output_figure(show=self._plot_settings['show'], file_output=self._plot_settings['file_output'])

    def get_regions(self):
        return deepcopy(self._regions)

    def get_outlines(self):
        return deepcopy(self._outlines)

    def get_grids(self):
        return deepcopy(self._grids)

    def get_points(self):
        return deepcopy(self._points)

    def get_dataset(self):
        return self.main_dataset


def update_figure_plotly(fig: Figure, updates: dict = None, **kwargs):
    """Updates the given Plotly figure with the 'geos','traces', and 'layout'.

    These update dict should look as follows:
    {
        'geos': { updates for geos (calls update_geos()) },
        'traces': { updates for traces (calls update_traces()) },
        'layout': { updates for layout (calls update_layout()) }
    }

    :param fig: The figure to update
    :type fig: Figure
    :param updates: The updates to add to the figure
    :type updates: dict
    :return: The updated figure
    :rtype: Figure
    """
    prop = kwargs
    if updates:
        prop.update(updates)

    fig.update_geos(prop.pop('geos', {}))
    fig.update(**prop)
