from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly
from plotly.graph_objects import Figure, Choropleth, Scattergeo

from . import hexgrid as gcg
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

import logging
from copy import deepcopy


pd.options.mode.chained_assignment = None

SHAPE_PATH = pjoin(pjoin(sys.path[0], 'data'), 'shapefiles')
CSV_PATH = pjoin(pjoin(sys.path[0], 'data'), 'csv-data')

region_definitions = {
    'CanadaCart': pjoin(pjoin(SHAPE_PATH, 'canada_definitions'), 'cartographic'),
    'CanadaBound': pjoin(pjoin(SHAPE_PATH, 'canada_definitions'), 'sample3-stations-canboundary'),
    'CanadaCoast': pjoin(pjoin(SHAPE_PATH, 'canada_definitions'), 'coastal_waters'),
    'CanadaAdmin': pjoin(pjoin(SHAPE_PATH, 'canada_definitions'), 'sample1-fires-canoutline')
}

sample_definitions = {
    'SAR_Bases': pjoin(CSV_PATH, 'sample4-sarbases.csv'),
    'SAR_Incidents': pjoin(CSV_PATH, 'sample3-sarincidents.csv'),
    'Canadian_AirPorts': pjoin(CSV_PATH, 'sample1-fires-canoutline.csv'),
    'Canadian_COVID': pjoin(CSV_PATH, 'sample2-covidcompiled.csv'),
    'Sample_Data': pjoin(pjoin(SHAPE_PATH, 'sample1-fires-canoutline'), 'sample1-fires-canoutline.shp')
}

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

'''
with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("default")



    for wi in w:
        if wi.line is None:
            wi.line = linecache.getline(wi.filename, wi.lineno)
        print('line number {}:'.format(wi.lineno))
        print('line: {}'.format(wi.line))
'''


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


def _get_hover_field(gdf: GeoDataFrame, hover_fields: dict):
    """Makes a text field for the given dataframe.

    :param gdf: The dataframe to make a text field for
    :type gdf: GeoDataFrame
    :param hover_fields: A dictionary including all labels and fields to go into the text field
    :type hover_fields: dict
    :return: A text field for the dataframe
    :rtype: GeoSeries
    """
    gdf['GEN-txt'] = ''
    for hov_field in hover_fields.items():
        gdf['GEN-txt'] += hov_field[0]
        gdf['GEN-txt'] += ' : '
        gdf['GEN-txt'] += gdf[hov_field[1]].astype(str)
        gdf['GEN-txt'] += '<br>'
    return gdf['GEN-txt']


_extension_mapping = {
    '.csv': pd.read_csv,
    '.xlsx': pd.read_excel,
    '.shp': gpd.read_file,
    '': gpd.read_file
}


def _convert(dataset: DataSet) -> DataSet:
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


def _convert_to_hex_dataset(dataset: DataSet, hex_resolution: int) -> DataSet:
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

    _convert(dataset)

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

    g_field = gcg.get_column_or_default(dataset['data'], 'binning_field')
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

        self._managers = {
            "dataset": {},
            "region": {
                "singular": {},
                "individual": defaultdict(dict)
            },
            "outline": {
                "singular": {},
                "individual": defaultdict(dict)
            },
            "point": {
                "singular": {},
                "individual": defaultdict(dict)
            },
            "grid": {},
            "figure": {},
            "file_output": {}
        }

        # managers and settings
        self._dataset_manager: DataSetManager = {}
        self._grid_manager: DataSetManager = {}
        self._figure_manager: FigureManager = {}
        self._file_output_manager: FileOutputManager = deepcopy(self._default_file_output_manager)
        self._plot_settings: DataSetManager = deepcopy(self._default_plot_settings)

        self.hex_resolution: int = self._default_hex_resolution
        self.range_buffer_lat: Tuple[float, float] = (0.0, 0.0)
        self.range_buffer_lon: Tuple[float, float] = (0.0, 0.0)
        self._hone_geometry: List[Union[Point, Polygon]] = []

        if logging_file:
            loggingAddFile(fix_filepath(logging_file, add_filename=self.plot_name or 'temp', add_ext='log'))
            toggleLogging(True)
            logger.debug('logger set.')

        self._figure: Figure = Figure()
        logger.debug('initialized internal figure.')

        self._datasets: DataSets = {}
        self._grids: DataSets = {}
        self._points: DataSets = {}
        self._regions: DataSets = {}
        self._outlines: DataSets = {}

        if main_dataset is not None:
            self.main_dataset = main_dataset
            if self._main_dataset['v_type'] == 'str':
                self._dataset_manager['colorscale'] = 'Set3'

        if grids is None:
            grids = {}
            if default_grids:
                if not _is_list_like(default_grids) or len(default_grids) != 2:
                    grids = self._default_grids.copy()
                else:
                    if default_grids[0]:
                        grids = self._default_grids.copy()
                        for key, val in grids.items():
                            grids[key]['hex_resolution'] = default_grids[1]

        for grid in grids.items():
            logger.debug(f'began loading grid, name={grid[0]}.')
            self.add_grid(grid[0], grid[1])

        input_regions = {}
        if default_regions:
            input_regions = self._default_regions.copy()
        if regions:
            input_regions.update(regions)

        for region in input_regions.items():
            logger.debug(f'began loading a region, name={region[0]}.')
            self.add_region(region[0], region[1])

        if outlines:
            for outline in outlines.items():
                logger.debug(f'began loading an outline, name={outline[0]}.')
                self.add_outline(outline[0], outline[1])

        if points:
            for point in points.items():
                logger.debug(f'began loading a set of points, name={point[0]}.')
                self.add_point(point[0], point[1])

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
        return deepcopy(self._datasets['*MAIN*'])

    def _get_main_dataset(self):
        return self._datasets['*MAIN*']

    @main_dataset.setter
    def main_dataset(self, value: DataSet):
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
        _convert_to_hex_dataset(value, self.hex_resolution)
        logger.debug('ended conversion of main dataset.')
        self._datasets['*MAIN*'] = value

    def add_grid(self, name: str, grid: DataSet):
        """Adds a region hex to the builder.

        :param name: The name this dataset will be referred to as
        :type name: str
        :param grid: The grid dataset to add
        :type grid: DataSet
        """

        logger.debug(f'began conversion of grid, name={name}.')
        try:
            grid['data'] = butil.get_shapes_from_world(grid['data'])
        except (KeyError, ValueError, TypeError):
            logger.debug("If a region name was a country or continent, the process failed.")

        _convert_to_hex_dataset(grid, self.hex_resolution)
        df = grid['data']

        df['value_field'] = 0
        grid['data'] = df

        self._grids[name] = grid
        logger.debug(f'ended conversion of grid, name={name}.')

    def add_region(self, name: str, region: DataSet):
        """Adds a region to the builder. (Should be loaded as GeoDataFrames)

        The region being added MUST have Polygon geometry.

        :param name: The name of the region hex to add
        :type name: str
        :param region: The dictionary as shown above
        :type region: DataSet
        """
        logger.debug(f'began conversion of region, name={name}.')

        try:
            df = butil.get_shapes_from_world(region['data'])
        except (KeyError, ValueError, TypeError):
            logger.debug("If a region name was a country or continent, the process failed.")
            _convert(region)
            df = region['data']

        df = gcg.conform_geogeometry(df, fix_polys=region.get('validate', True))[['geometry']]

        if region.get('to_boundary', False):
            geom = df.unary_union.boundary
            polys = [Polygon(list(g.coords)) for g in geom]
            mpl = MultiPolygon(polys)

            df = GeoDataFrame({'geometry': [x := mpl]}, crs='EPSG:4326')

        df['text_field'] = name
        df['value_field'] = 0

        logger.debug('dataframe geometry conformed to GeoJSON standard.')
        region['data'] = df

        manager = {}
        dict_deep_update(manager, region.get('manager', {}))
        region['manager'] = manager

        self._regions[name] = region
        logger.debug(f'ended conversion of region, name={name}.')

    def add_outline(self, name: str, outline: DataSet):
        tempname = f'*{name}*'
        self.add_region(tempname, outline)
        outline = self._regions.pop(tempname)
        self._outlines[name] = outline

    def add_region_from_world(self, name, store_as: Optional[str] = None):
        region_dataset = {'data': butil.get_shapes_from_world(name)}
        store_as = store_as if store_as else name
        self.add_region(store_as, region_dataset)

    def add_point(self, name: str, point: DataSet):
        """Adds a point hex to the builder.

        :param name: The name this dataset will be referred to as
        :type name: str
        :param point: The point dataset to add
        :type point: DataSet
        """
        logger.debug(f'began conversion of points, name={name}.')
        _convert(point)
        try:
            df = point['data'].explode()
        except IndexError:
            df = point['data']

        manager = {}
        dict_deep_update(manager, point.get('manager', {}))
        point['manager'] = manager
        point['data'] = df
        self._points[name] = point
        logger.debug(f'ended conversion of points, name={name}.')

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

        cen = gcg.find_center(self._hone_geometry)
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
        print('[MAIN DATASET]\n', self._datasets['*MAIN*'], '\n')
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

        geojson = gcg.geodataframe_to_geojson(gdf, 'value_field')

        choro = Choropleth(
            locations=gdf.index,
            z=gdf['value_field'],
            geojson=geojson,
            legendgroup='choros'
        ).update(initial_properties).update(final_properties if final_properties else {})

        logger.debug('choropleth trace generated.')
        return choro

    def _add_generic_traces(self, gdf: GeoDataFrame, **kwargs):
        """Adds a generic choropleth trace to the internal figure.

        :param gdf: The GeoDataFrame to plot
        :type gdf: GeoDataFrame
        :param kwargs: Keyword arguments to update the plotly choropleth with
        :type kwargs: **kwargs
        """

        # gdf = gcg.conform_geodataframe_geometry(gdf)
        geojson = gcg.geodataframe_to_geojson(gdf, 'value_field')

        choro = Choropleth(
            locations=gdf.index,
            z=gdf['value_field'],
            geojson=geojson,
            legendgroup='choros'
        ).update(**kwargs)

        self._figure.add_trace(choro)
        logger.debug('choropleth trace added to plot.')

    def _get_auto_grid(self) -> DataSet:
        """Gets the auto grid dataset.

        :return: The auto grid dataset
        :rtype: DataSet
        """

        clip_mode = self._plot_settings['clip_mode']
        if self._plot_settings['auto_grid']:
            logger.debug('attempting to get auto grid dataset.')
            if clip_mode == 'regions':
                grid = gcg.get_hex_geodataframe_loss(self._regions['*COMBINED*']['data'])

            elif clip_mode == 'outlines':
                grid = gcg.get_hex_geodataframe_loss(self._outlines['*COMBINED*']['data'])

            elif self._datasets['*MAIN*']:
                grid = gcg.generate_grid_over_hexes(self._datasets['*MAIN*']['data'])

            else:
                raise gce.BuilderDatasetInfoError("There is no dataset to form an auto-grid with!")

            return {
                'data': grid.rename({'value': 'value_field'}, axis=1, errors='raise')
            }
        raise gce.BuilderPlotBuildError("Auto grid is not enabled!")

    def _make_merged_grid_dataset(self) -> DataSet:

        if self._plot_settings['plot_grids'] or self._plot_settings['clip_mode'] == 'grids':
            try:
                self._grids['*AUTO*'] = self._get_auto_grid()
            except gce.BuilderPlotBuildError:
                pass

            self._grids['*COMBINED*'] = {
                'data': gcg.merge_datasets_simple([ds['data'] for ds in list(self._grids.values())]).rename(
                    {'merge-op': 'value_field'}, axis=1, errors='raise')}
            logger.debug('merged grid datasets.')

    def _get_grid_dataset(self, additional_grids: Optional[List[DataSet]] = None,
                          clip_regions: Optional[GeoDataFrame] = None) -> DataSet:
        """Gets the merged grid dataset.

        :param additional_grids: Additional grid datasets to combine with the ones currently in the builder
        :type additional_grids: Optional[List[DataSet]]
        :param clip_regions: A dataframe to clip the auto grid to if it is on
        :type clip_regions: Optional[GeoDataFrame]
        :return: Merged grid dataset
        :rtype: DataSet
        """

        grids_on = self._plot_settings['plot_grids']
        clip_mode = self._plot_settings['clip_mode']
        auto_grid = self._plot_settings['auto_grid']
        grids = list(self._grids.values())

        if additional_grids is not None and self._plot_settings['replot_empties']:
            grids.extend(additional_grids)

        if len(grids) > 0 or auto_grid:

            if clip_mode == 'grids' or grids_on:

                try:
                    logger.debug('attempting to get auto grid dataset.')
                    if clip_mode == 'regions' and clip_regions is not None:
                        auto_gridded = self._get_auto_grid(clip_regions=clip_regions)
                    else:
                        auto_gridded = self._get_auto_grid()
                    if isvalid_dataset(auto_gridded):
                        logger.debug('auto grid dataset added.')
                        grids.append(auto_gridded)
                except gce.BuilderPlotBuildError:
                    pass
                if len(grids) > 0:
                    to_merge = [grid['data'] for grid in grids]
                    frame = gcg.merge_datasets_simple(to_merge, drop=True, crs='EPSG:4326')
                    frame.rename({'merge-op': 'value_field'}, axis=1, inplace=True, errors='raise')
                    logger.debug('merged grid datasets.')
                    return dict(
                        data=frame
                    )
            else:
                return None
        else:
            raise gce.BuilderPlotBuildError("There were no grid-type datasets detected!")

    def _clip_to_regions(self):
        try:
            rds = self._regions['*COMBINED*']
            if self._plot_settings['clip_mode'] == 'regions':
                try:
                    ds = self._datasets['*ALTERED*']
                    ds['data'] = gcg.clip_hexes_to_polygons(ds['data'], rds['data'])
                except (KeyError, AttributeError):
                    pass

                try:
                    gds = self._grids['*COMBINED*']
                    gds['data'] = gcg.clip_hexes_to_polygons(gds['data'], rds['data'])
                except (KeyError, AttributeError):
                    pass

                if self._plot_settings['clip_points']:
                    try:
                        pds = self._points['*COMBINED*']
                        pds['data'] = gcg.clip_points_to_polygons(pds['data'], rds['data'])
                    except (KeyError, AttributeError):
                        pass
        except KeyError:
            pass

    def _clip_to_outlines(self):
        try:
            ods = self._outlines['*COMBINED*']
            if self._plot_settings['clip_mode'] == 'outlines':
                try:
                    ds = self._datasets['*ALTERED*']
                    ds['data'] = gcg.clip_hexes_to_polygons(ds['data'], ods['data'])
                except (KeyError, AttributeError):
                    pass

                try:
                    gds = self._grids['*COMBINED*']
                    gds['data'] = gcg.clip_hexes_to_polygons(gds['data'], ods['data'])
                except (KeyError, AttributeError):
                    pass

                if self._plot_settings['clip_points']:
                    try:
                        pds = self._points['*COMBINED*']
                        pds['data'] = gcg.clip_points_to_polygons(pds['data'], ods['data'])
                    except (KeyError, AttributeError):
                        pass
        except KeyError:
            pass

    def _clip_to_grids(self):
        try:
            gds = self._grids['*COMBINED*']
            if self._plot_settings['clip_mode'] == 'grids':
                try:
                    ds = self._datasets['*ALTERED*']
                    ds['data'] = gcg.clip_hexes_to_polygons(ds['data'], gds['data'])
                except (KeyError, AttributeError):
                    pass

                if self._plot_settings['clip_points']:
                    try:
                        pds = self._points['*COMBINED*']
                        pds['data'] = gcg.clip_points_to_polygons(pds['data'], gds['data'])
                    except (KeyError, AttributeError):
                        pass
        except KeyError:
            pass

    def _clip_datasets(self):
        """Clips the geometries of each merged dataset.

        :param ds: The main dataset
        :type ds: DataSet
        :param rds: The merged region dataset
        :type rds: DataSet
        :param gds: The merged grid dataset
        :type gds: DataSet
        :param pds: The merged point dataset
        :type pds: DataSet
        """

        clip_mode = self._plot_settings['clip_mode']

        self._clip_to_regions()
        self._clip_to_grids()
        self._clip_to_outlines()

        '''
                if clip_mode == 'regions' and isvalid_dataset(rds):
            if isvalid_dataset(ds):
                ds['data'] = gcg.clip_hexes_to_polygons(ds['data'], rds['data'])

            if isvalid_dataset(gds):
                gds['data'] = gcg.clip_hexes_to_polygons(gds['data'], rds['data'])
                grids_clipped = True
            regions_clipped = True

        elif clip_mode == 'grids' and isvalid_dataset(gds):
            if isvalid_dataset(ds):
                ds['data'] = gcg.clip_hexes_to_hexes(ds['data'], gds['data'])

            grids_clipped = True

        elif clip_mode == 'outlines' and isvalid_dataset(ods):
            if isvalid_dataset(ds):
                ds['data'] = gcg.clip_hexes_to_polygons(ds['data'], ods['data'])

            if isvalid_dataset(gds):
                gds['data'] = gcg.clip_hexes_to_polygons(gds['data'], ods['data'])
                grids_clipped = True
            outlines_clipped = True

        if clip_points and isvalid_dataset(pds):
            if isvalid_dataset(ds):
                pds['data'] = gcg.clip_points_to_polygons(pds['data'], ds['data'])
            else:
                pds['data'] = GeoDataFrame(DataFrame(columns=pds['data'].columns), geometry=None, crs='EPSG:4326')

            pdgeom = GeoDataFrame(pds['data'].geometry.to_frame(name='geometry'), geometry='geometry')
            pdgeom.crs = pds['data'].crs

            if regions_clipped:
                pdf2 = gcg.clip_points_to_polygons(pdgeom, rds['data'])
                pds['data'] = pds['data'].merge(pdf2['geometry'], how='outer')

            if grids_clipped:
                pdf1 = gcg.clip_points_to_polygons(pdgeom, gds['data'])
                pds['data'] = pds['data'].merge(pdf1['geometry'], how='outer')

            if outlines_clipped:
                pdf3 = gcg.clip_points_to_polygons(pdgeom, ods['data'])
                pds['data'] = pds['data'].merge(pdf3['geometry'], how='outer')
        '''

        logger.debug(f'plot datasets clipped, mode={clip_mode}.')

    def _prepare_dataset(self):

        ds = self.main_dataset  # KeyError thrown here if not found
        if isvalid_dataset(ds):
            df = ds['data']
            v_type = ds['v_type']
            empty_symbol = 'empty' if v_type == 'str' else 0
            empties = df[df['value_field'] == empty_symbol]
            if len(empties) > 0:
                empties['value_field'] = 0
                if self._plot_settings['replot_empties']:
                    self._grids['*REMOVED*'] = {'data': empties}

            if self._plot_settings['remove_empties']:
                df = df[df['value_field'] != empty_symbol]
                logger.debug(f'removed empty rows from main dataset, length={len(df)}.')

            cpy = deepcopy(ds)
            cpy.pop('data')
            self._datasets['*ALTERED*'] = {'data': df, **cpy}
        else:
            raise gce.BuilderPlotBuildError("The main dataset was invalid.")

    def _get_dataset(self):
        """Gets the main dataset and alters it slightly.

        :return: The main dataset (or None)
        :rtype: DataSet
        """

        if isvalid_dataset(self._datasets['*MAIN*']):

            ds = self._datasets['*MAIN*']
            df = ds['data']
            v_type = ds['v_type']

            if not df.empty:

                empty_symbol = 'empty' if v_type == 'str' else 0
                empties = df[df['value_field'] == empty_symbol]
                if len(empties) > 0:
                    empties['value_field'] = 0
                    if self._plot_settings['replot_empties']:
                        self._grids['*REMOVED*'] = {'data': empties}

                if self._plot_settings['remove_empties']:
                    df = df[df['value_field'] != empty_symbol]
                    logger.debug(f'removed empty rows from main dataset, length={len(df)}.')

                ds['data'] = df

                return ds
            return None
        raise gce.BuilderPlotBuildError("The main dataset was not valid.")

    def _make_dataset_traces(self, ds: DataSet):

        try:
            df = ds['data']
        except KeyError:
            pass

    def _plot_dataset_traces(self):
        """Plots the trace of the main dataset onto the internal figure.

        :param ds: The main dataset (after alteration)
        :type ds: DataSet
        """

        dgeoms = []
        try:
            ds = self._datasets['*ALTERED*']
            self._output_stats['main-dataset'] = defaultdict(dict)
            logger.debug('adding main dataset to plot.')
            df = ds['data']

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

            sm = self._plot_settings.pop('scale_mode', cs.pop('scale_mode', 'linear'))
            if isinstance(sm, dict):
                mode = sm.pop('mode').lower()
                kw = sm
            else:
                mode = sm.lower()
                kw = {}

            if mode == 'logarithmic':
                butil.logify_scale(DataFrame({'value_field': []}))  # remove this

                updates = butil.logify_scale(df, **kw)
                dict_deep_update(self._dataset_manager, updates)
                bounds = updates['zmin'], updates['zmax']

            plot_dfs = []
            if ds['v_type'] == 'num':
                df_prop = deepcopy(self._dataset_manager)
                logger.debug(f'scale implemented, mode={mode}.')
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
        except KeyError:
            pass
        return dgeoms

    def _remove_underlying_grid(self, ds: DataSet, gds: DataSet):
        """Removes the grid from underneath the data (lower file size).

        :param ds: The main dataset (after altreration)
        :type ds: DataSet
        :param gds: The merged grid dataset (after alteration)
        :type gds: DataSet
        """

        if isvalid_dataset(ds):
            df = ds['data']
            gdf = gds['data']
            gdf = gpd.overlay(gdf, df[['value_field', 'geometry']],
                              how='difference')
            gdf = gcg.remove_other_geometries(gdf, 'Polygon')
            gds['data'] = gdf

    def _plot_regions(self):
        rgeoms = []

        if self._plot_settings['plot_regions']:
            logger.debug('adding regions to plot.')
            try:
                rds = self._regions['*COMBINED*']
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
            except KeyError:
                pass
        return rgeoms

    def _plot_outlines(self):

        ogeoms = []
        if self._plot_settings['plot_outlines']:
            logger.debug('adding outlines to plot.')
            try:
                ods = self._outlines['*COMBINED*']
                self._output_stats['outlines'] = {}
                initial_prop = deepcopy(self._default_outline_manager)
                for outname, outds in dissolve_multi_dataset(ods, self._outlines).items():
                    self._output_stats['outlines'][outname] = get_stats(outds['data'])
                    initial_prop['name'] = outname
                    initial_prop['text'] = f'{outname}-outline'
                    self._figure.add_trace(
                        self._make_general_scatter_trace(outds['data'], initial_prop, final_properties=outds['manager']))
                ogeoms = list(ods['data'].geometry)
            except KeyError:
                pass
        return ogeoms

    def _plot_grids(self):

        ggeoms = []
        if self._plot_settings['plot_grids']:
            logger.debug('adding grids to plot.')
            try:
                gds = self._grids['*COMBINED*']
                logger.debug('adding grids to plot.')
                try:
                    self._remove_underlying_grid(self._datasets['*ALTERED*'], gds)
                except KeyError:
                    pass

                gds['data']['text_field'] = 'GRID'
                self._output_stats['grids'] = get_stats(gds['data'], hexed=True)
                initial_prop = deepcopy(self._default_grid_manager)
                initial_prop['text'] = 'GRID'
                print(gds['data'])
                self._figure.add_trace(
                    x:=self._make_general_trace(gds['data'], initial_prop, final_properties=self._grid_manager))
                print(x)
                ggeoms = list(gds['data'].geometry)
            except KeyError as e:
                raise e
                pass
        return ggeoms

    def _plot_points(self):
        pgeoms = []
        if self._plot_settings['plot_grids']:
            logger.debug('adding points to plot.')
            try:
                pds = self._points['*COMBINED*']
                self._output_stats['points'] = {}
                initial_prop = deepcopy(self._default_point_manager)
                for poiname, poids in dissolve_multi_dataset(pds, self._points).items():
                    self._output_stats['points'][poiname] = get_stats(poids['data'])
                    initial_prop['name'] = poiname
                    initial_prop['hovertext'] = _get_hover_field(poids['data'], poids['hover_fields'])
                    initial_prop['text'] = get_column_or_default(poids['data'], 'text_field')
                    self._figure.add_trace(
                        self._make_general_scatter_trace(poids['data'], initial_prop, final_properties=poids['manager'],
                                                         disjoint=False))
                pgeoms = list(pds['data'].geometry)
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

        regions_on = self._plot_settings['plot_regions']
        grids_on = self._plot_settings['plot_grids']
        outlines_on = self._plot_settings['plot_outlines']
        points_on = self._plot_settings['plot_points']

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

        self.update_plot_settings(**kwargs)

        try:
            self._prepare_dataset()
        except KeyError:
            logger.warning('No main dataset was detected. Ignoring and returning figure...')
        except gce.BuilderPlotBuildError:
            logger.warning('The main dataset was invalid. Ingoring and returning figure...')

        '''
        try:
            rds = self._get_region_dataset()
        except gce.BuilderPlotBuildError as e:
            logger.warning(f'{e.message} Ignoring and returning figure...')
            rds = None
        '''

        try:
            self._make_multi_region_dataset()
        except ValueError:
            logger.warning('No region-type datasets were detected. Ignoring and returning figure...')

        try:
            self._make_multi_outline_dataset()
        except ValueError:
            logger.warning('No outline-type datasets were detected. Ignoring and returning figure...')

        try:
            self._make_multi_point_dataset()
        except ValueError:
            logger.warning('No point-type datasets were detected. Ignoring and returning figure...')

        try:
            self._make_merged_grid_dataset()
        except IndexError:
            logger.warning('No grid-type datasets were detected. Ignoring and returning figure...')


        self._clip_datasets()
        self._plot_traces()
        #self._plot_traces()

        update_figure_plotly(self._figure, self._default_figure_manager)
        update_figure_plotly(self._figure, self._figure_manager)
        self._focus()

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
