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

from geopandas import GeoDataFrame

from geoviz import builderrevised as builder
from shapely.geometry import Point
from shapely import wkt
from testingstructures import TestingShape
from pandas import DataFrame
from os.path import join as pjoin
import os

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
CSV_PATH = pjoin(DATA_PATH, 'csv-data')
SHAPE_PATH = pjoin(DATA_PATH, 'shapefiles')

testpoints = [
    'POINT(-196.171875 47.040182144806664)',
    'POINT(-135 56.17002298293205)',
    'POINT(-144.140625 52.908902047770255)',
    'POINT(-130.078125 33.7243396617476)',
    'POINT(-157.5 39.90973623453719)',
    'POINT(-175.78125 33.137551192346145)',
    'POINT(-143.4375 24.5271348225978)',
    'POINT(-202.5 27.059125784374068)',
    'POINT(-93.515625 31.952162238024975)',
    'POINT(-137.109375 23.885837699862005)',
    'POINT(-139.21874999999997 25.799891182088334)',
    'POINT(-145.546875 29.53522956294847)',
    'POINT(-140.2734375 24.5271348225978)',
    'POINT(-174.0234375 33.43144133557529)',
    'POINT(-156.97265625 39.90973623453719)',
    'POINT(-139.5703125 29.6880527498568)',
    'POINT(-143.08593749999997 28.304380682962783)',
    'POINT(-140.2734375 29.84064389983441)',
    'POINT(-143.61328125 53.12040528310657)',
    'POINT(-144.66796875 52.908902047770255)',
    'POINT(-143.96484375 53.85252660044951)',
    'POINT(-133.06640625 56.9449741808516)',
    'POINT(-195.29296875 47.040182144806664)',
    'POINT(-202.5 26.745610382199022)'
]


class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.builder = builder.PlotBuilder()

    def test_set_main_dataset_quantitative(self):
        """Tests quantitative dataset functionality.

        Sets the main dataset to a quantitative dataset,
        then checks if the result is correct.
        """
        vg = [(10, wkt.loads(p)) for p in testpoints]
        vals, geoms = zip(*vg)
        testdf = GeoDataFrame(dict(val=vals, geometry=geoms), crs='EPSG:4326')

        self.builder.set_main(testdf, binning_field='val', binning_fn='min')

        getmain = self.builder._get_main()  # internal version does not return a deepcopy
        self.assertTrue(getmain)  # test to see if the main dataset was added correctly
        # ensure the information stored in the dataset is valid
        self.assertTrue('data' in getmain)
        self.assertTrue('manager' in getmain)
        self.assertTrue('value_field' in getmain['data'].columns)
        self.assertEqual(getmain['DSTYPE'], 'MN')
        self.assertEqual(getmain['VTYPE'], 'num')
        # ensure the main dataframe does not reference the dame input dataframe
        self.assertFalse(testdf.equals(getmain['data']))
        # ensure the dataframe was binned properly
        self.assertTrue(all(val == 10 for val in getmain['data']['value_field']))


    def test_set_main_dataset_qualitative(self):
        geoms = [wkt.loads(p) for p in testpoints]
        vals = ['EVENDS' if x % 2 == 0 else 'ODDDS' for x in range(len(geoms))]
        testdf = GeoDataFrame(dict(val=vals, geometry=geoms), crs='EPSG:4326')
        self.builder.set_main(
            testdf,
            binning_field='val',
            manager=dict(
                colorscale={
                    'EVENDS': 'red',
                    'ODDDS': 'yellow'
                }
            )
        )
        getmain = self.builder._get_main()  # internal version does not return a deepcopy
        self.assertTrue(getmain)  # test to see if the main dataset was added correctly
        # ensure the information stored in the dataset is valid
        self.assertTrue('data' in getmain)
        self.assertTrue('manager' in getmain)
        self.assertTrue('value_field' in getmain['data'].columns)
        self.assertEqual(getmain['DSTYPE'], 'MN')
        self.assertEqual(getmain['VTYPE'], 'str')
        # ensure the main dataframe does not reference the dame input dataframe
        self.assertFalse(testdf.equals(getmain['data']))
        self.builder.build_plot(raise_errors=False)
        self.builder.display_figure()

    def test_add_main_dataset(self):
        self.builder.set_main(
            pjoin(CSV_PATH, 'sample5-arrivalanalysis.csv'),
            latitude_field='latitude',
            longitude_field='longitude',
            binning_field='arrival_diff',
            binning_fn=min
        )

        self.builder.add_region('CCA1', 'CANADA')

        self.builder.update_main_manager(
            colorscale='Viridis',
            marker=dict(
                opacity=0.8
            ),
            colorbar=dict(title='Arrival Diff')
        )
        self.builder.adjust_focus(validate=False)
        self.builder.adjust_opacity()
        # self.builder.simple_clip(method='sjoin')
        self.builder.build_plot(raise_errors=False)
        self.builder.display_figure()
        self.builder.output_figure('E:/software/nicholiv1.pdf', clear_figure=True)

        self.builder.set_main(
            pjoin(CSV_PATH, 'sample5-arrivalanalysis.csv'),
            latitude_field='latitude',
            longitude_field='longitude',
            binning_field='max_loiter',
            binning_fn=max
        )

        self.builder.add_region('CCA1', 'CANADA')

        self.builder.update_main_manager(
            colorscale='Viridis',
            marker=dict(
                opacity=0.8
            ),
            colorbar=dict(title='Max Loiter')
        )
        self.builder.adjust_focus(validate=False)
        self.builder.adjust_opacity()
        # self.builder.simple_clip(method='sjoin')
        self.builder.build_plot(raise_errors=False)
        self.builder.display_figure()
        self.builder.output_figure('E:/software/nicholiv2.pdf', clear_figure=True)

        self.builder.set_main(
            pjoin(CSV_PATH, 'sample5-arrivalanalysis.csv'),
            latitude_field='latitude',
            longitude_field='longitude',
            binning_field='mission_time',
            binning_fn=max
        )

        self.builder.add_region('CCA1', 'CANADA')

        self.builder.update_main_manager(
            colorscale='Viridis',
            marker=dict(
                opacity=0.8
            ),
            colorbar=dict(title='Mission Time (Max)')
        )
        self.builder.adjust_focus(validate=False)
        self.builder.adjust_opacity()
        # self.builder.simple_clip(method='sjoin')
        self.builder.build_plot(raise_errors=False)
        self.builder.display_figure()
        self.builder.output_figure('E:/software/nicholiv3.pdf', clear_figure=True)

    def test_quals(self):
        self.builder.set_main(
            pjoin(CSV_PATH, 'sample5-arrivalanalysis.csv'),
            latitude_field='latitude',
            longitude_field='longitude',
            binning_field='class',
            binning_fn="bestworst"
        )

        self.builder.add_region('CCA1', 'CANADA')

        self.builder.update_main_manager(
            colorscale='Set3',
            marker=dict(
                opacity=0.4
            ),
            colorbar=dict(title='Max Loiter')
        )

        self.builder.add_point('CCA2',
                               pjoin(CSV_PATH, 'sample4-sarbases.csv'),
                               latitude_field='latitude',
                               longitude_field='longitude'
                               )

        self.builder.auto_focus(validate=False)
        self.builder.opacify_colorscale()
        self.builder.remove_point('CCA2')
        self.builder.build_plot(raise_errors=False)
        self.builder.update_figure(layout=dict(paper_bgcolor='rgb(0,0,0)'))
        self.builder.display_figure()
        self.builder.output_figure('E:/software/nicholiv2.pdf', clear_figure=True)

    def test_add_regions(self):
        maindf = DataFrame({
            'lat': [45, 60, 60, 70, 80, 0],
            'lon': [45, 60, 60, 70, 80, 0],
            'sval': ['DS1', 'DS1', 'DS2', 'DS2', 'DS2', 'DS3'],
            'nval': [10, 10, 10, 20, 20, 20]
        })

        self.builder.set_main(pjoin(CSV_PATH, 'sample3-sarincidents.csv'),
                              latitude_field='incpos_latitude', longitude_field='incpos_longitude')

        self.builder.add_region('CCA1', 'CANADA')
        self.builder.logify_scale()
        self.builder.auto_focus('region:CCA1')
        self.builder.discretize_scale(discrete_size=1, choose_hues=[1, 3, 5, 7, -1])
        self.builder.build_plot(raise_errors=False)

        self.builder.output_figure('E:\software\hellothere2.pdf', clear_figure=False)
        self.builder.display_figure()


if __name__ == '__main__':
    unittest.main()
