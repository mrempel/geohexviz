"""
Notes for refactoring:

We need to monitor the process of the builder, externally.
We have access to the attributes.

Think about testing:
1) Adding the main data set (and what happens to it)
2) Adding the regions, points, grids, and outlines (and what happen to them)
3) Plot output (file exists?)
4) Logging file
"""

import unittest
from geoviz import builderrev as builder
from shapely.geometry import Point
from testingstructures import TestingShape
from pandas import DataFrame
from os.path import join as pjoin
import os

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
CSV_PATH = pjoin(DATA_PATH, 'csv-data')
SHAPE_PATH = pjoin(DATA_PATH, 'shapefiles')

testpoints1 = TestingShape(Point(45, 12), Point(60, 12), Point(60, 30), condense=False)

class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.builder = builder.PlotBuilder(default_grids=True)

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
        self.builder.build_plot(show=True, scale_mode='logarithmic', clip_mode=None)
        print(self.builder.main_dataset)

    def test_c(self):
        self.assertTrue(True)

        self.builder.main_dataset = {
            'data': pjoin(CSV_PATH, 'sample3-sarincidents.csv'),
            'latitude_field': 'incpos_latitude',
            'longitude_field': 'incpos_longitude'
        }

        self.builder.add_region('CCA', {
            'data': 'CANADA'
        })

        self.builder.add_grid('CCA', {
            'data': 'CANADA'
        })

        self.builder.add_grid('CCA2', {
            'data': 'FRANCE'
        })



        self.builder.add_point('SAR_BASES', {
            'data': pjoin(CSV_PATH, 'sample4-sarbases.csv'),
            'latitude_field': 'latitude',
            'longitude_field': 'longitude',
            'manager': {
                'marker': {
                    'size': 20
                }
            }
        })
        self.builder.add_outline('NEW1', {
            'data': 'SOUTH AMERICA',
            'to_boundary': True
        })

        self.builder.clip_datasets(clip_mode='regions+outlines')
        #self.builder.clip_datasets(clip_mode='regions')




        self.builder.print_datasets()

        self.builder.build_plot(show=True, plot_points=True, clip_points=True)



if __name__ == '__main__':
    unittest.main()
