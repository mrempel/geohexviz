from plotly.graph_objects import Scattergeo, Choropleth
from geopandas import GeoDataFrame
from ..hexgrid import geodataframe_to_geojson
from ..utils.plot_util import to_plotly_points_format
from copy import deepcopy



class DataContainer:
    def __init__(self, data: GeoDataFrame, properties=None, **kwargs):
        kwargs.update(properties if properties else {})
        self.odata = data
        self.adata = None
        self.reset_data()
        self._properties = {}
        self.properties = kwargs

    @property
    def data(self):
        return self.get_data()

    def get_data(self):
        return self.adata

    @data.setter
    def data(self, data):
        self.set_data(data)

    def set_data(self, data: GeoDataFrame):
        self.adata = data

    @property
    def properties(self):
        return self.get_properties()

    def get_properties(self):
        return deepcopy(self.properties)

    @properties.setter
    def properties(self, properties: dict):
        self.set_properties(properties)

    def set_properties(self, properties: dict):
        self.properties = deepcopy(properties)

    def reset_data(self):
        self.data = self.odata.copy(deep=True)


class PlotlyDataContainer(DataContainer):

    def __init__(self, data: GeoDataFrame, traceclass, properties=None, **kwargs):
        super().__init__(data, properties=properties, **kwargs)
        self.tracer = traceclass()
        if isinstance(self.tracer, Choropleth):
            self.updater = self.ChoroplethUpdater()
        elif isinstance(self.tracer, Scattergeo):
            self.updater = self.ScattergeoUpdater()
        else:
            raise NotImplementedError("The updater for the class you specified has not been implemented.")

    def prepare_trace(self, baseonly=True, overwrite=False):
        self.updater.update_trace(self.tracer, self.adata, overwrite=overwrite)
        if not baseonly:
            self.tracer.update(overwrite=overwrite, **self.properties)

    class ChoroplethUpdater:
        def update_trace(self, choro: Choropleth, data: GeoDataFrame, **kwargs):
            geojson = geodataframe_to_geojson(data, value_field='value_field')
            choro.update(geojson=geojson, z=data['value_field'].astype(float), **kwargs)

    class ScattergeoUpdater:
        def update_trace(self, scatt: Scattergeo, data: GeoDataFrame, **kwargs):
            lats, lons = to_plotly_points_format(data)
            scatt.update(lat=lats, lon=lons, **kwargs)
