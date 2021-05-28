import unittest
from geoviz import builder
from shapely.geometry import Point, Polygon
from testingstructures import TestingShape
from geopandas import GeoDataFrame
from pandas import DataFrame

testpoints1 = TestingShape(Point(45, 12), Point(60, 12), Point(60, 30), condense=False)

class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.builder = builder.PlotBuilder()

    def test_add_main_dataset(self):
        maindf = DataFrame({
            'lat': [45, 60, 60, 70, 80, 0],
            'lon': [45, 60, 60, 70, 80, 0],
            'val': [1, 2, 3, 4, 5, 6]
        })

        self.builder.main_dataset = {
            'data': maindf,
            'latitude_field': 'lat',
            'longitude_field': 'lon',
            'binning_field': 'val',
            'binning_fn': lambda lst: sum(lst)**2
        }

        ds = self.builder.main_dataset
        #testbuilder.build_plot(show=True)
        print(self.builder.main_dataset)

    def test_add_region(self):
        self.builder.add_region()

if __name__ == '__main__':
    unittest.main()
