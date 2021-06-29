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
import math
import unittest

from geopandas import GeoDataFrame
import geopandas as gpd

from geoviz import builderrevised as builder
from shapely import wkt
from pandas import DataFrame
from os.path import join as pjoin
import os
from test_geoutils import shapes_from_wkt
import numpy as np

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
CSV_PATH = pjoin(DATA_PATH, 'csv-data')
SHAPE_PATH = pjoin(DATA_PATH, 'shapefiles')


class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.builder = builder.PlotBuilder()

    def test_set_main_dataset_quantitative(self):
        """Tests quantitative dataset functionality.

        Sets the main dataset to a quantitative dataset,
        then checks if the result is correct.
        """
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

    def test_add_region(self):
        self.builder.add_region('RRA1', 'CANADA')
        getreg = self.builder._get_region('RRA1')
        self.assertEqual(getreg['DSTYPE'], 'RGN')
        self.assertTrue('value_field' in getreg['data'])
        self.assertTrue(all(v == 0 for v in getreg['data']['value_field']))

        self.builder.add_region('RRA2', pjoin(SHAPE_PATH, pjoin('polygon-like', 'sample2-canbuffer')))
        getreg = self.builder._get_region('RRA2')
        self.assertEqual(getreg['DSTYPE'], 'RGN')
        self.assertTrue('value_field' in getreg['data'])
        self.assertTrue(all(v == 0 for v in getreg['data']['value_field']))

    def test_add_grid(self):
        self.builder.add_grid('GGA1', 'FRANCE')
        getgr = self.builder._get_grid('GGA1')
        self.assertEqual(getgr['DSTYPE'], 'GRD')
        self.assertTrue('value_field' in getgr['data'])
        self.assertTrue(all(v == 0 for v in getgr['data']['value_field']))

        self.builder.add_grid('GGA2', pjoin(SHAPE_PATH, pjoin('polygon-like', 'sample2-canbuffer')))
        getgr = self.builder._get_grid('GGA2')
        self.assertEqual(getgr['DSTYPE'], 'GRD')
        self.assertTrue('value_field' in getgr['data'])
        self.assertTrue(all(v == 0 for v in getgr['data']['value_field']))

    def test_add_outline(self):
        self.builder.add_outline('OOA1', 'SOUTH AMERICA')
        getout = self.builder._get_outline('OOA1')
        self.assertEqual(getout['DSTYPE'], 'OUT')
        self.assertTrue('value_field' in getout['data'])
        self.assertTrue(all(v == 0 for v in getout['data']['value_field']))

        self.builder.add_outline('OOA2', pjoin(SHAPE_PATH, pjoin('polygon-like', 'sample2-canbuffer')))
        getout = self.builder._get_outline('OOA2')
        self.assertEqual(getout['DSTYPE'], 'OUT')
        self.assertTrue('value_field' in getout['data'])
        self.assertTrue(all(v == 0 for v in getout['data']['value_field']))

    def test_add_points(self):
        testpoints = shapes_from_wkt(*[
            'POINT (10.0634765625 51.26191485308451)',
            'POINT (10.283203125 51.01375465718821)',
            'POINT (18.193359375 53.4357192066942)',
            'POINT (18.7646484375 51.86292391360244)',
            'POINT (13.842773437499998 56.84897198026975)',
            'POINT (21.8408203125 53.12040528310657)',
            'POINT (20.0830078125 53.330872983017066)',
            'POINT (19.9951171875 51.01375465718821)',
            'POINT (8.5693359375 51.37178037591737)',
            'POINT (10.2392578125 52.9883372533954)',
            'POINT (10.6787109375 50.00773901463687)',
            'POINT (8.0419921875 50.17689812200107)',
            'POINT (3.9111328125000004 48.45835188280866)',
            'POINT (2.8125 46.92025531537451)',
            'POINT (12.7001953125 42.45588764197166)',
            'POINT (24.5654296875 46.07323062540835)'
        ])

        inputdf = GeoDataFrame(geometry=testpoints, crs='EPSG:4326')
        self.builder.add_point('PPA1', inputdf)
        getpnt = self.builder._get_point('PPA1')
        self.assertEqual(getpnt['DSTYPE'], 'PNT')
        self.assertTrue('value_field' in getpnt['data'])
        self.assertTrue(all(v == 0 for v in getpnt['data']['value_field']))

    def test_clipping(self):
        """Test the builder's ability to clip datasets to other datasets.

        Lengthier runtime.
        Tests:
        Clip datasets within the builder to each other. Ensure the resulting
        data length matches that which was expected. Clip method does not
        really matter here.
        """

        # 5 points outside of Poland and Germany, 10 points inside.
        # this generates 5 hexes in Germany, 5 in Poland, and 5 outside
        testdata = shapes_from_wkt(*[
            'POINT (3.9111328125000004 48.45835188280866)',
            'POINT (2.8125 46.92025531537451)',
            'POINT (12.7001953125 42.45588764197166)',
            'POINT (24.5654296875 46.07323062540835)',
            'POINT (16.875 53.657661020298)',
            'POINT (20.98388671875 54.059387886623576)',
            'POINT (22.8076171875 52.06600028274635)',
            'POINT (19.62158203125 50.387507803003146)',
            'POINT (8.525390625 48.07807894349862)',
            'POINT (6.83349609375 50.56928286558243)',
            'POINT (8.72314453125 53.14677033085082)',
            'POINT (5.185546875 57.088515327886505)',
            'POINT (17.0947265625 51.984880139916626)',
            'POINT (10.30517578125 50.12057809796008)',
            'POINT (10.92041015625 52.10650519075632)'
        ])

        self.builder.set_main(GeoDataFrame(geometry=testdata, crs='EPSG:4326'))

        # add regions for Germany and Poland
        self.builder.add_region('RRA1', 'GERMANY')
        self.builder.add_region('RRA2', 'POLAND')

        getmain = self.builder._get_main()
        initiallen = len(getmain['data'])

        # test clipping main to the first region alone
        self.builder.clip_datasets('main', 'region:RRA1')
        newlen = len(self.builder._get_main()['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(5, newlen)  # for this dataset

        # test clipping main to the second region alone
        self.builder.reset_main_data()  # reset back to original for another clip
        self.builder.clip_datasets('main', 'region:RRA2')
        newlen = len(getmain['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(5, newlen)  # for this dataset

        # test clipping main to all regions
        self.builder.reset_main_data()
        self.builder.clip_datasets('main', 'regions')
        newlen = len(getmain['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(10, newlen)  # for this dataset

        # add a grid over europe
        self.builder.add_grid('GGA1', 'EUROPE')
        # add an outline over france and italy
        self.builder.add_outline('OOA1', 'FRANCE')
        self.builder.add_outline('OOA2', 'ITALY')

        # test clipping grids to main
        self.builder.reset_main_data()
        gridlen = sum(self.builder.apply_to_query('grids', lambda dataset: len(dataset['data'])))
        self.builder.clip_datasets('grids', 'main', operation='within')
        newgridlen = sum(self.builder.apply_to_query('grids', lambda dataset: len(dataset['data'])))
        self.assertLess(newgridlen, gridlen)  # for this dataset
        self.assertEqual(14, newgridlen)  # for this dataset

        # test clipping main to grids
        self.builder.reset_main_data()
        self.builder.clip_datasets('main', 'grids', operation='within')  # intersects will get a different result
        newlen = len(getmain['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(14, newlen)  # for this dataset

        # test clipping main to the first outline alone
        self.builder.reset_main_data()
        self.builder.clip_datasets('main', 'outline:OOA1')
        newlen = len(getmain['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(2, newlen)  # for this dataset

        # test clipping main to the second outline alone
        self.builder.reset_main_data()
        self.builder.clip_datasets('main', 'outline:OOA2')
        newlen = len(getmain['data'])
        self.assertLess(newlen, initiallen)  # for this dataset
        self.assertEqual(1, newlen)  # for this dataset

        # test clipping main to all outlines
        self.builder.reset_main_data()
        self.builder.clip_datasets('main', 'outlines', method='gpd')
        newlen = len(getmain['data'])
        self.assertEqual(newlen, initiallen)  # for this dataset
        self.assertEqual(3, newlen)  # for this dataset

        self.builder.reset_main_data()
        outlen = sum(self.builder.apply_to_query('outlines', lambda dataset: len(dataset['data'])))
        self.builder.clip_datasets('outlines', 'main', operation='intersects')
        newoutlen = sum(self.builder.apply_to_query('outlines', lambda dataset: len(dataset['data'])))
        self.assertLessEqual(newoutlen, outlen)
        self.assertEqual(2, newoutlen)  # for this dataset

        # test clipping main to points, which does not make sense
        self.builder.reset_main_data()
        self.builder.add_point('PPA1', GeoDataFrame(geometry=testdata, crs='EPSG:4326'))

        err = False
        try:
            self.builder.clip_datasets('main', 'points', method='gpd')
        except TypeError:
            err = True
        self.assertTrue(err)

    def test_logify_scale(self):
        """Tests the builder's ability to make a logarithmic scale.

        This test also tests most of the logify_info() functionality (in plot_util).

        Tests:
        Set the main dataset and invoke the function. Ensure that the dataset's
        value_field has been changed and it's manager contains accurate Plotly properties.
        """

        testdata = shapes_from_wkt(*[
            'POINT (3.9111328125000004 48.45835188280866)',
            'POINT (2.8125 46.92025531537451)',
            'POINT (12.7001953125 42.45588764197166)',
            'POINT (24.5654296875 46.07323062540835)',
            'POINT (16.875 53.657661020298)',
            'POINT (20.98388671875 54.059387886623576)',
            'POINT (22.8076171875 52.06600028274635)',
            'POINT (19.62158203125 50.387507803003146)',
            'POINT (8.525390625 48.07807894349862)',
            'POINT (6.83349609375 50.56928286558243)',
            'POINT (8.72314453125 53.14677033085082)',
            'POINT (5.185546875 57.088515327886505)',
            'POINT (17.0947265625 51.984880139916626)',
            'POINT (10.30517578125 50.12057809796008)',
            'POINT (10.92041015625 52.10650519075632)'
        ])

        import random

        values = [random.randint(1, 100000) for i in range(len(testdata))]

        self.builder.set_main(GeoDataFrame(dict(geometry=testdata, val=values), crs='EPSG:4326'),
                              binning_fn=min, binning_field='val')

        getmain = self.builder._get_main()
        oldvals = getmain['data']['value_field']
        self.builder.logify_scale()
        newvals = getmain['data']['value_field']
        nvalmax, nvalmin = max(newvals), min(newvals)
        start, end = int(math.floor(nvalmin)), int(math.floor(nvalmax))+1

        # ensure the values are logged
        self.assertTrue(all(y == np.log10(x) for x, y in zip(oldvals, newvals)))

        # ensure all necessary values are in colorbar properties
        self.assertTrue(all(x in getmain['manager']['colorbar']['tickvals'] for x in range(start, end)))


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

    def test_quick(self):
        data = gpd.read_file(pjoin(CSV_PATH, 'sample3-sarincidents.csv'))

        import geoviz.utils.geoutils as gcg
        data = gcg.convert_dataframe_coordinates_to_geodataframe(data)
        data = gcg.hexify_geodataframe(data, 3)
        import time
        start = time.time()
        tstdf = gcg.bin_by_hex_withgeoms(data, lambda lst: len(lst))
        end = time.time()
        print(f'DIFF {end - start}')
        start = time.time()
        tstdf2 = gcg.bin_by_hex(data, lambda lst: len(lst))
        end = time.time()
        print(f'DIFF {end - start}')

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

        self.builder.adjust_focus(validate=False)
        self.builder.adjust_opacity()
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
