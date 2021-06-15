from typing import Any, Tuple

import geopandas as gpd
import pandas as pd
from os import path
from os.path import join as pjoin
from geoviz.utils.util import fix_filepath, get_sorted_occurrences, generate_dataframe_random_ids, get_column_type, \
    simplify_dicts
from geoviz.utils import geoutils as gcg
from geoviz.utils import plot_util as butil

from geopandas import GeoDataFrame
from pandas import DataFrame

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


def _read_dataset(dataset: dict, default_manager: dict = None):
    if default_manager is None:
        default_manager = {}

    try:
        data = dataset['data']
    except KeyError:
        raise ValueError("There must be a 'data' member in the dataset.")

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

    # set the manager
    input_manager = dataset.pop('manager', {})
    dataset['manager'] = default_manager
    dataset['manager'].update(input_manager)  # this changes to dict deep update later


def _hexify_data(data, hex_resolution: int):
    return gcg.hexify_geodataframe(data, hex_resolution=hex_resolution)


def _bin_by_hex(data, *args, binning_field: str = None, binning_fn=None, **kwargs):
    if binning_fn in _group_functions:
        binning_fn = _group_functions[binning_fn]

    if binning_field is None:
        vtype = 'num'
    else:
        vtype = get_column_type(data, 'binning_field')

    if binning_fn is None:
        binning_fn = _group_functions['bestworst'] if vtype == 'str' else _group_functions['count']

    return gcg.bin_by_hex(data, binning_fn, *args, binning_field=binning_field, result_name='value_field',
                          add_geoms=True, **kwargs), vtype


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


def _prepare_general_dataset(dataset: dict):
    try:
        dataset['data'] = butil.get_shapes_from_world(dataset['data'])
    except (KeyError, ValueError, TypeError):
        # logger.debug("If name was a country or continent, the process failed.")
        pass

    _read_dataset(dataset)
    dataset['data'] = gcg.conform_geogeometry(dataset['data'], fix_polys=True)[['geometry']]
    # logger.debug('dataframe geometry conformed to GeoJSON standard.')

    if dataset.pop('to_boundary', False):
        dataset['data'] = gcg.unify_geodataframe(dataset['data'])
    dataset['data']['value_field'] = 0
    dataset['VTYPE'] = 'num'


class PlotBuilder:

    def __init__(self, main=None, regions=None, grids=None, outlines=None, points=None):
        self._container = {
            'main': {},
            'regions': {},
            'grids': {},
            'outlines': {},
            'points': {}
        }

        # grids will all reference this manager
        self._grid_manager = {}

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

    def set_main(self, data, **kwargs):
        _read_dataset(dataset := _create_dataset(data, **kwargs), default_manager={})
        _hexbinify_dataset(dataset, 3)
        dataset['empties'] = GeoDataFrame()
        input_manager = dataset.pop('manager', {})
        self.update_main_manager(**input_manager)

        self._container['main']['sets']['MAIN'] = dataset
        print(self._container['main'])

    def get_main(self):
        try:
            return self._container['main']
        except KeyError:
            raise KeyError(f"The main dataset could not be found.")

    def get_main_data(self):
        return self.get_main()['sets']['MAIN']

    def remove_main(self):
        try:
            self._container['main'] = {'sets': {}, 'manager': {}}
        except KeyError:
            raise KeyError(f"The main dataset could not be found.")  # consider making this pass

    def update_main_manager(self, updates: dict = None, override: bool = False, **kwargs):
        if updates is None:
            updates = {}
        updates.update(kwargs)
        _update_helper(self._container['main'], updates, override)

    def reset_main(self):
        self._container['main'] = {}

    """
    REGION FUNCTIONS
    """

    def add_region(self, name: str, data, **kwargs):
        _prepare_general_dataset(dataset := _create_dataset(data, **kwargs))
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
            self.get_regions().pop(name)
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
        _prepare_general_dataset(dataset := _create_dataset(data, **kwargs))
        _hexbinify_dataset(dataset, 3)  # change to default hex res
        dataset['manager'] = self._grid_manager
        self.get_grids()['sets'][name] = dataset

    def get_grid(self, name: str):
        try:
            return self.get_grids()['sets'][name]
        except KeyError:
            raise KeyError(f"The grid-type dataset with the name '{name}' could not be found.")

    def get_grids(self):
        return self._container['grids']

    def remove_grid(self, name: str):
        try:
            grids = self.get_grids()
            grids.pop(name)
            if len(grids) == 0:
                self.reset_grids()
        except KeyError:
            raise ValueError(f"The grid-type dataset with the name '{name}' could not be found.")

    def update_grid_manager(self, updates: dict = None, override: bool = False, **kwargs):
        updates = simplify_dicts(fields=updates, **kwargs)
        _update_helper(self._grid_manager, updates, override)

    def reset_grids(self):
        self._container['grids'] = {}
        self._grid_manager = {}  # reset to default

    """
    OUTLINE FUNCTIONS
    """

    def add_outline(self, name: str, data, **kwargs):
        _read_dataset(dataset := _create_dataset(data, **kwargs))
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
            self.get_outlines().pop(name)
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
        _prepare_general_dataset(dataset := _create_dataset(data, **kwargs))
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
            self.get_points().pop(name)
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
    DATA ALTERING FUNCTIONS
    """

    def remove_empties(self, name: str = 'main', empty_symbol: Any = 0, add_to_plot: bool = False):
        datasets = self[name]

        print(datasets)
        if add_to_plot and name != 'main':
            raise ValueError('Empties can only be added to plot if it is the main dataset.')
        try:
            if add_to_plot:
                datasets['empties'].append(datasets['data'][datasets['data']['value_field'] == empty_symbol])
            datasets['data'] = datasets['data'][datasets['data']['value_field'] != empty_symbol]
        except KeyError:  # implies multiple datasets
            for ds in datasets.items():
                ds['data'] = ds['data'][ds['data']['value_field'] != empty_symbol]

    # this is both a data altering, and plot altering function
    def logify_scale(self, name: str, **kwargs):
        print()

    def clip_datasets(self, clip: str, to: str, op: str = 'intersects'):
        print()

    def simple_clip(self, to: str):
        print()

    """
    PLOT ALTERING FUNCTIONS
    """

    def opacify_colorscale(self, name: str = 'main'):
        datasets = self[name]

    def auto_focus(self, on: str = 'main'):
        datasets = self[on]
        try:
            geoms = list(datasets['data']['geometry'])
        except KeyError:
            geoms = []
            for ds in datasets.items():
                geoms.extend(list(ds['data']['geometry']))

    def auto_grid(self, name: str, by_bounds: bool = False):
        datasets = self[name]
        grid = GeoDataFrame()

    def discretize_scale(self, name: str, scale_type: str = 'sequential', **kwargs):
        print()

    """
    RETRIEVAL/SEARCHING FUNCTIONS
    """

    def search(self, query: str):
        sargs = query.split('+')
        resulting = {}
        for item in sargs:
            if item in ['regions', 'grids', 'outlines', 'points', 'main']:
                resulting[item] = self._container[item]
                continue

            try:
                typer, name = _split_name(item)
            except ValueError:
                raise ValueError("The given string should be one of ['regions', 'grids', 'outlines', 'points', "
                                 f"'main'] or in the form of '<type>:<name>'. Received item: {item}.")

            if typer == 'region':
                resulting[item] = self.get_region(name)
            elif typer == 'grid':
                resulting[item] = self.get_grid(name)
            elif typer == 'outline':
                resulting[item] = self.get_outline(name)
            elif typer == 'point':
                resulting[item] = self.get_point(name)
            else:
                raise ValueError(f"The given dataset type does not exist. Must be one of ['region', 'grid', "
                                 f"'outline', 'point']. "
                                 f"Received {typer}.")
        return resulting

    """
    PLOTTING FUNCTIONS
    """

    def plot_regions(self):
        print()

    def plot_grids(self):
        print()

    def plot_dataset(self):
        print()

    def plot_outlines(self):
        print()

    def plot_points(self):
        print()

    def build_plot(self):
        print()

    def output_figure(self, filepath: str, **kwargs):
        print()

    def display_figure(self, **kwargs):
        print()

    def reset(self):
        print()


if __name__ == '__main__':
    x = DataFrame(dict(lats=[10, 10, 20, 20, 30, 30], lons=[10, 10, 20, 20, 30, 30]))

    pb = PlotBuilder()
    pb.set_main(x, latitude_field='lats', longitude_field='lons', binning_fn=lambda x: 0,
                manager=dict(colorscale='Viridis'))
    pb.add_region('tony', data=x, latitude_field='lats', longitude_field='lons')
    pb.update_managers('regions', colorscale='Inferno')

    pb['region:nottony'] = dict(
        data=x, latitude_field='lats', longitude_field='lons'
    )
    print(pb.search_data_dict('regions+main'))
    # pb.remove_empties('main')

    # dataset = _read_dataset(x, latitude_field='lats', longitude_field='lons', binning_fn=dict(fn=sum))
    # _hexbinify_dataset(dataset, 3)
    # print(dataset['data']['value_field'])
