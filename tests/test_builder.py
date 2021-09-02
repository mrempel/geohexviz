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
from shapely.geometry import Polygon

from geohexviz import builder as builder
from geohexviz import errors as err
from shapely import wkt
from pandas import DataFrame
from os.path import join as pjoin
import os
from tests.testutils.test_geoutils import shapes_from_wkt
import numpy as np

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
CSV_PATH = pjoin(DATA_PATH, 'csv-data')
SHAPE_PATH = pjoin(DATA_PATH, 'shapefiles')


def alterer(dataset):
    dataset['data']['value_field'] = -1


# TODO: see how long it will take to get new model implemented.
class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.builder = builder.PlotBuilder()

    def test_quick(self):
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
        self.builder.set_main(testdf, hexbin_info=dict(hex_resolution=3, binning_fn=min, binning_field='val'))
        self.builder.add_region('CCA1', 'CANADA')
        self.builder.add_grid('GGA1', 'FRANCE')
        self.builder.add_outline('OOA1', 'SOUTH AMERICA')
        self.builder.adjust_focus('region:CCA1')
        self.builder.adjust_colorbar_size()

        testdf = GeoDataFrame(dict(val=vals, geometry=geoms), crs='EPSG:4326')
        self.builder.add_point('PPA1', testdf)
        # self.builder.clip_datasets('main', 'region:CCA1')
        self.builder.build_plot()

        self.builder.output_figure('E:\\software\\testing.pdf')
        self.builder.display_figure()

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

        self.builder.set_main(testdf, hexbin_info=dict(binning_field='val', binning_fn=min))

        getmain = self.builder._get_main()  # internal version does not return a deepcopy
        self.assertTrue(getmain)  # test to see if the main dataset was added correctly
        # ensure the information stored in the dataset is valid
        self.assertTrue('data' in getmain)
        self.assertTrue('manager' in getmain)
        self.assertTrue('value_field' in getmain['data'].columns)
        self.assertEqual(getmain['DSTYPE'], 'MN')
        self.assertEqual(getmain['VTYPE'], 'NUM')
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
            hexbin_info=dict(
                binning_field='val'
            ),
            manager=dict(
                colorscale=['blue', 'pink']
            )
        )
        getmain = self.builder._get_main()  # internal version does not return a deepcopy
        self.assertTrue(getmain)  # test to see if the main dataset was added correctly
        # ensure the information stored in the dataset is valid
        self.assertTrue('data' in getmain)
        self.assertTrue('manager' in getmain)
        self.assertTrue('value_field' in getmain['data'].columns)
        self.assertEqual(getmain['DSTYPE'], 'MN')
        self.assertEqual(getmain['VTYPE'], 'STR')
        # ensure the main dataframe does not reference the dame input dataframe
        self.assertFalse(testdf.equals(getmain['data']))
        self.builder.build_plot()
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
        self.assertLess(newlen, initiallen)  # for this dataset
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
                              hexbin_info=dict(binning_fn=min, binning_field='val'))

        getmain = self.builder._get_main()
        oldvals = getmain['data']['value_field']
        self.builder.logify_scale()
        newvals = getmain['data']['value_field']
        nvalmax, nvalmin = max(newvals), min(newvals)
        start, end = int(math.floor(nvalmin)), int(math.floor(nvalmax)) + 1

        # ensure the values are logged
        self.assertTrue(all(y == np.log10(x) for x, y in zip(oldvals, newvals)))

        # ensure all necessary values are in colorbar properties
        self.assertTrue(all(x in getmain['manager']['colorbar']['tickvals'] for x in range(start, end)))

    def test_adjust_opacity(self):
        """Tests the adjusting opacity feature of the builder.

        Tests:
        Set the main dataset with a custom colorscale and invoke the function.
        Ensure the opacity is present within the colors of the colorscale.
        """
        with self.assertRaises(err.MainDatasetNotFoundError):
            self.builder.adjust_opacity()

        inp_colorscale = [[0, 'rgb(10, 10, 10)'], [0.5, 'rgb(50, 50, 50)'], [1, 'rgb(90, 90, 90)']]
        inp_opacity = 0.6
        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )),
            manager=dict(
                colorscale=inp_colorscale,
                marker=dict(opacity=inp_opacity)
            )
        )
        self.builder.adjust_opacity()
        out_colorscale = self.builder._get_main()['manager']['colorscale']
        self.assertTrue(all(str(inp_opacity) in ci for _, ci in out_colorscale))

    def test_discretize_scale(self):
        """Tests the builder's ability to make a discrete scale.

        Tests:
        Set the main dataset alongside a custom colorscale and invoke the function.
        Ensure that the colors are present twice in the output colorscale.
        """
        with self.assertRaises(err.MainDatasetNotFoundError):
            self.builder.discretize_scale()

        inp_colorscale = [[0, 'rgb(10, 10, 10)'], [0.5, 'rgb(50, 50, 50)'], [1, 'rgb(90, 90, 90)']]
        # inp_colorscale = ['red', 'blue', 'green']
        inp_ds = 1
        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )),
            manager=dict(
                zmin=1,
                colorscale=inp_colorscale
            )
        )
        getmain = self.builder._get_main()
        self.builder.discretize_scale(discrete_size=inp_ds)
        out_colorscale = getmain['manager']['colorscale']
        inpcolors = [ci for _, ci in inp_colorscale]
        outcolors = [ci for _, ci in out_colorscale]
        self.assertTrue(all(outcolors.count(ci) == 2 for ci in inpcolors))

    def test_adjust_focus(self):
        """Tests the builders ability to adjust the focus of the plot.

        Tests:
        Add a dataset to the builder and focus onto it.
        Ensure that the geospatial properties have been set properly.
        """
        self.builder.add_point(
            'PPA1',
            df := DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            ))
        )
        self.builder.adjust_focus(on='point:PPA1')
        rlat, rlon = [min(df['latitude']), max(df['latitude'])], [min(df['latitude']), max(df['latitude'])]
        getgeo = self.builder._figure.layout.geo
        self.assertListEqual(rlat, list(getgeo.lataxis.range))
        self.assertListEqual(rlon, list(getgeo.lonaxis.range))

        testpoly = Polygon([[1, 1], [1, 10], [10, 10], [10, 1], [1, 1]])
        self.builder.add_region(
            'RRA1',
            GeoDataFrame(
                geometry=[testpoly]
            )
        )
        self.builder.adjust_focus(on='region:RRA1')
        # from bounds may be of use with auto grid
        minlon, minlat, maxlon, maxlat = testpoly.bounds
        rlat, rlon = [minlat, maxlat], [minlon, maxlon]
        self.assertListEqual(rlat, list(getgeo.lataxis.range))
        self.assertListEqual(rlon, list(getgeo.lonaxis.range))

    def test_apply_to_query(self):
        """Tests the builder's ability to apply a function to a query

        Tests:
        Add datasets and apply a query to them. Ensure the resulting
        datasets have been altered in the correct way.
        """
        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        getmain = self.builder._get_main()
        self.builder.apply_to_query('all', alterer)
        self.assertTrue(all(x == -1 for x in getmain['data']['value_field'].values))

    def test_update_main_manager(self):
        """Tests the builder's ability to update the main dataset's manager.

        Tests:
        Set the main dataset and update it. Check if it was updated properly.
        """
        with self.assertRaises(err.MainDatasetNotFoundError):
            self.builder.update_main_manager(colorscale='Viridis')
        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        self.builder.update_main_manager(colorscale='Picnic')
        self.assertEqual('Picnic', self.builder._get_main()['manager']['colorscale'])


    def test_clear_main_manager(self):
        """Tests the builder's ability to clear the manager of the main dataset.

        Tests:
        Set the main dataset and clear it's manager. Ensure the manager is empty.
        """
        with self.assertRaises(err.MainDatasetNotFoundError):
            self.builder.clear_main_manager()
        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        self.builder.clear_main_manager()
        self.assertEqual({}, self.builder._get_main()['manager'])

    def test_reset_main_data(self):
        """Tests the builder's ability to reset the main dataset's data back to its original state.

        Tests:
        Set the main dataset. Alter it's data and reset it. Ensure that the data is equal to the original data.
        """
        with self.assertRaises(err.MainDatasetNotFoundError):
            self.builder.reset_main_data()

        self.builder.set_main(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))

        getmain = self.builder._get_main()
        vals = getmain['data']['value_field'].copy()
        self.builder.apply_to_query('main', alterer)
        self.assertTrue(all(x == -1 for x in getmain['data']['value_field'].values))
        self.builder.reset_main_data()
        self.assertTrue(getmain['data']['value_field'].equals(vals))

    def test_update_region_manager(self):
        """Tests the builder's ability to update the manager of region datasets.

        Tests:
        Add region datasets and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.DatasetNotFoundError):
            self.builder.update_region_manager(name='RRA1', colorscale='Picnic')
        self.builder.update_region_manager(colorscale='Picnic')  # may change this behaviour (this does nothing)
        self.builder.add_region('RRA1', 'CANADA')
        self.builder.add_region('RRA2', 'FRANCE')

        rra1 = self.builder._get_region('RRA1')
        rra2 = self.builder._get_region('RRA2')

        self.builder.update_region_manager(name='RRA1', colorscale='Inferno')
        self.assertEqual('Inferno', rra1['manager']['colorscale'])
        self.assertNotEqual('Inferno', rra2['manager']['colorscale'])

        self.builder.update_region_manager(colorscale='Plasma')
        self.assertEqual('Plasma', rra1['manager']['colorscale'])
        self.assertEqual('Plasma', rra2['manager']['colorscale'])

        self.builder.update_region_manager(overwrite=True, legendgroup='regions')
        with self.assertRaises(KeyError):
            _ = rra1['manager']['colorscale']

    def test_clear_region_manager(self):
        """Tests the builder's ability to clear the manager of region datasets.

        Tests:
        Add region datasets and clear their managers. Ensure they are empty.
        """
        self.builder.clear_region_manager()
        self.builder.add_region('RRA1', 'CANADA')
        self.builder.add_region('RRA2', 'FRANCE')

        rra1 = self.builder._get_region('RRA1')
        rra2 = self.builder._get_region('RRA2')

        self.assertNotEqual({}, rra1['manager'])
        self.assertNotEqual({}, rra2['manager'])
        self.builder.clear_region_manager()

        self.assertEqual({}, rra1['manager'])
        self.assertEqual({}, rra2['manager'])

    def test_reset_region_data(self):
        """Tests the builder's ability to reset the data of region datasets.

        Tests:
        Add region datasets and alter their data. Revert the data. Ensure they are back to original state.
        """
        self.builder.add_region('RRA1', 'CANADA')
        self.builder.add_region('RRA2', 'FRANCE')

        rra1 = self.builder._get_region('RRA1')
        rra2 = self.builder._get_region('RRA2')

        rra1vals = rra1['data']['value_field'].copy()
        rra2vals = rra2['data']['value_field'].copy()

        self.builder.apply_to_query('regions', alterer)

        self.assertTrue(all(x == -1 for x in rra1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in rra2['data']['value_field'].values))
        self.builder.reset_region_data(name='RRA1')
        self.assertTrue(rra1['data']['value_field'].equals(rra1vals))
        self.assertFalse(rra2['data']['value_field'].equals(rra2vals))

        self.builder.apply_to_query('regions', alterer)
        self.assertTrue(all(x == -1 for x in rra1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in rra2['data']['value_field'].values))
        self.builder.reset_region_data(name='RRA2')
        self.assertFalse(rra1['data']['value_field'].equals(rra1vals))
        self.assertTrue(rra2['data']['value_field'].equals(rra2vals))

        self.builder.apply_to_query('regions', alterer)
        self.assertTrue(all(x == -1 for x in rra1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in rra2['data']['value_field'].values))
        self.builder.reset_region_data()
        self.assertTrue(rra1['data']['value_field'].equals(rra1vals))
        self.assertTrue(rra2['data']['value_field'].equals(rra2vals))

    def test_update_grid_manager(self):
        """Tests the builder's ability to update the manager of grid datasets.

        Tests:
        Add grid datasets and update their managers. Ensure they were updated correctly.
        """
        self.builder.update_grid_manager(colorscale='Viridis')
        self.builder.add_grid('GGA1', 'CANADA')
        self.builder.add_grid('GGA2', 'FRANCE')

        gga1 = self.builder._get_grid('GGA1')
        gga2 = self.builder._get_grid('GGA2')

        self.assertNotEqual('Viridis', gga1['manager']['colorscale'])
        self.assertNotEqual('Viridis', gga2['manager']['colorscale'])
        self.assertEqual(gga1['manager'], gga2['manager'])

        self.builder.update_grid_manager(colorscale='Picnic')
        self.assertEqual('Picnic', gga1['manager']['colorscale'])
        self.assertEqual('Picnic', gga2['manager']['colorscale'])
        self.assertEqual(gga1['manager'], gga2['manager'])

        self.builder.update_grid_manager(overwrite=True, legendgroup='grids')
        with self.assertRaises(KeyError):
            _ = gga1['manager']['colorscale']

    def test_clear_grid_manager(self):
        """Tests the builder's ability to clear the manager of grid datasets.

        Tests:
        Add grid datasets and clear their managers. Ensure they are empty.
        """
        self.builder.clear_grid_manager()
        self.builder.add_grid('GGA1', 'CANADA')
        self.builder.add_grid('GGA2', 'FRANCE')

        gga1 = self.builder._get_grid('GGA1')
        gga2 = self.builder._get_grid('GGA2')

        self.assertNotEqual({}, gga1['manager'])
        self.assertNotEqual({}, gga2['manager'])
        self.assertEqual(gga1['manager'], gga2['manager'])
        self.builder.clear_grid_manager()

        self.assertEqual({}, gga1['manager'])
        self.assertEqual({}, gga2['manager'])
        self.assertEqual(gga1['manager'], gga2['manager'])

    def test_reset_grid_data(self):
        """Tests the builder's ability to reset the data of grid datasets.

        Tests:
        Add grid datasets and alter their data. Revert the data. Ensure they are back to original state.
        """
        self.builder.add_grid('GGA1', 'CANADA')
        self.builder.add_grid('GGA2', 'FRANCE')

        gga1 = self.builder._get_grid('GGA1')
        gga2 = self.builder._get_grid('GGA2')

        gga1vals = gga1['data']['value_field'].copy()
        gga2vals = gga2['data']['value_field'].copy()

        self.builder.apply_to_query('grids', alterer)
        self.assertTrue(all(x == -1 for x in gga1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in gga2['data']['value_field'].values))
        self.builder.reset_grid_data(name='GGA1')
        self.assertTrue(gga1['data']['value_field'].equals(gga1vals))
        self.assertFalse(gga2['data']['value_field'].equals(gga2vals))

        self.builder.apply_to_query('grids', alterer)
        self.assertTrue(all(x == -1 for x in gga1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in gga2['data']['value_field'].values))
        self.builder.reset_grid_data(name='GGA2')
        self.assertFalse(gga1['data']['value_field'].equals(gga1vals))
        self.assertTrue(gga2['data']['value_field'].equals(gga2vals))

        self.builder.apply_to_query('grids', alterer)
        self.assertTrue(all(x == -1 for x in gga1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in gga2['data']['value_field'].values))
        self.builder.reset_grid_data()
        self.assertTrue(gga1['data']['value_field'].equals(gga1vals))
        self.assertTrue(gga2['data']['value_field'].equals(gga2vals))

    def test_update_outline_manager(self):
        """Tests the builder's ability to update the manager of outline datasets.

        Tests:
        Add outline datasets and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.DatasetNotFoundError):
            self.builder.update_outline_manager(name='OOA1', mode='markers')
        self.builder.update_outline_manager(mode='markers')  # may change this behaviour (this does nothing)
        self.builder.add_outline('OOA1', 'CANADA')
        self.builder.add_outline('OOA2', 'FRANCE')

        ooa1 = self.builder._get_outline('OOA1')
        ooa2 = self.builder._get_outline('OOA2')

        self.builder.update_outline_manager(name='OOA1', mode='markers')
        self.assertEqual('markers', ooa1['manager']['mode'])
        self.assertNotEqual('markers', ooa2['manager']['mode'])

        self.builder.update_outline_manager(mode='lines+markers')
        self.assertEqual('lines+markers', ooa1['manager']['mode'])
        self.assertEqual('lines+markers', ooa2['manager']['mode'])

        self.builder.update_outline_manager(overwrite=True, legendgroup='outlines')
        with self.assertRaises(KeyError):
            _ = ooa1['manager']['mode']

    def test_clear_outline_manager(self):
        """Tests the builder's ability to clear the manager of outline datasets.

        Tests:
        Add outline datasets and clear their managers. Ensure they are empty.
        """
        self.builder.clear_outline_manager()
        self.builder.add_outline('OOA1', 'CANADA')
        self.builder.add_outline('OOA2', 'FRANCE')

        ooa1 = self.builder._get_outline('OOA1')
        ooa2 = self.builder._get_outline('OOA2')

        self.assertNotEqual({}, ooa1['manager'])
        self.assertNotEqual({}, ooa2['manager'])
        self.builder.clear_outline_manager()

        self.assertEqual({}, ooa1['manager'])
        self.assertEqual({}, ooa2['manager'])

    def test_reset_outline_data(self):
        """Tests the builder's ability to reset the data of outline datasets.

        Tests:
        Add outline datasets and alter their data. Revert the data. Ensure they are back to original state.
        """
        self.builder.add_outline('OOA1', 'CANADA')
        self.builder.add_outline('OOA2', 'FRANCE')

        ooa1 = self.builder._get_outline('OOA1')
        ooa2 = self.builder._get_outline('OOA2')

        ooa1vals = ooa1['data']['value_field'].copy()
        ooa2vals = ooa2['data']['value_field'].copy()

        self.builder.apply_to_query('outlines', alterer)
        self.assertTrue(all(x == -1 for x in ooa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ooa2['data']['value_field'].values))
        self.builder.reset_outline_data(name='OOA1')
        self.assertTrue(ooa1['data']['value_field'].equals(ooa1vals))
        self.assertFalse(ooa2['data']['value_field'].equals(ooa2vals))

        self.builder.apply_to_query('outlines', alterer)
        self.assertTrue(all(x == -1 for x in ooa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ooa2['data']['value_field'].values))
        self.builder.reset_outline_data(name='OOA2')
        self.assertFalse(ooa1['data']['value_field'].equals(ooa1vals))
        self.assertTrue(ooa2['data']['value_field'].equals(ooa2vals))

        self.builder.apply_to_query('outlines', alterer)
        self.assertTrue(all(x == -1 for x in ooa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ooa2['data']['value_field'].values))
        self.builder.reset_outline_data()
        self.assertTrue(ooa1['data']['value_field'].equals(ooa1vals))
        self.assertTrue(ooa2['data']['value_field'].equals(ooa2vals))

    def test_update_point_manager(self):
        """Tests the builder's ability to update the manager of point datasets.

        Tests:
        Add point datasets and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.DatasetNotFoundError):
            self.builder.update_point_manager(name='PPA1', mode='markers')
        self.builder.update_point_manager(mode='markers')  # may change this behaviour (this does nothing)
        self.builder.add_point('PPA1', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))
        self.builder.add_point('PPA2', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))

        ppa1 = self.builder._get_point('PPA1')
        ppa2 = self.builder._get_point('PPA2')

        self.builder.update_point_manager(name='PPA1', mode='markers')
        self.assertEqual('markers', ppa1['manager']['mode'])
        self.assertNotEqual('markers', ppa2['manager']['mode'])

        self.builder.update_point_manager(mode='lines+markers')
        self.assertEqual('lines+markers', ppa1['manager']['mode'])
        self.assertEqual('lines+markers', ppa2['manager']['mode'])

        self.builder.update_point_manager(overwrite=True, legendgroup='points')
        with self.assertRaises(KeyError):
            _ = ppa1['manager']['mode']

    def test_clear_point_manager(self):
        """Tests the builder's ability to clear the manager of point datasets.

        Tests:
        Add point datasets and clear their managers. Ensure they are empty.
        """
        self.builder.clear_point_manager()
        self.builder.add_point('PPA1', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))
        self.builder.add_point('PPA2', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))

        ppa1 = self.builder._get_point('PPA1')
        ppa2 = self.builder._get_point('PPA2')

        self.assertNotEqual({}, ppa1['manager'])
        self.assertNotEqual({}, ppa2['manager'])
        self.builder.clear_point_manager()

        self.assertEqual({}, ppa1['manager'])
        self.assertEqual({}, ppa2['manager'])

    def test_reset_point_data(self):
        """Tests the builder's ability to reset the data of point datasets.

        Tests:
        Add point datasets and alter their data. Revert the data. Ensure they are back to original state.
        """

        self.builder.add_point('PPA1', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))
        self.builder.add_point('PPA2', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))

        ppa1 = self.builder._get_point('PPA1')
        ppa2 = self.builder._get_point('PPA2')

        ppa1vals = ppa1['data']['value_field'].copy()
        ppa2vals = ppa2['data']['value_field'].copy()

        self.builder.apply_to_query('points', alterer)
        self.assertTrue(all(x == -1 for x in ppa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ppa2['data']['value_field'].values))
        self.builder.reset_point_data(name='PPA1')
        self.assertTrue(ppa1['data']['value_field'].equals(ppa1vals))
        self.assertFalse(ppa2['data']['value_field'].equals(ppa2vals))

        self.builder.apply_to_query('points', alterer)
        self.assertTrue(all(x == -1 for x in ppa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ppa2['data']['value_field'].values))
        self.builder.reset_point_data(name='PPA2')
        self.assertFalse(ppa1['data']['value_field'].equals(ppa1vals))
        self.assertTrue(ppa2['data']['value_field'].equals(ppa2vals))

        self.builder.apply_to_query('points', alterer)
        self.assertTrue(all(x == -1 for x in ppa1['data']['value_field'].values))
        self.assertTrue(all(x == -1 for x in ppa2['data']['value_field'].values))
        self.builder.reset_point_data()
        self.assertTrue(ppa1['data']['value_field'].equals(ppa1vals))
        self.assertTrue(ppa2['data']['value_field'].equals(ppa2vals))

    def test_reset(self):
        print()

        self.builder.add_grid('GGA1', 'CANADA')
        self.builder.add_outline('OOA1', 'CANADA')
        self.builder.add_point('PPA1', DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        )))

    def test_auto_grid(self):
        """Tests the builders ability to... may scrap.

        :return:
        :rtype:
        """
        # need to test by bounds version
        self.builder.add_point(
            'PPA1',
            df := DataFrame(dict(
                latitude=[10, 20, 30],
                longitude=[10, 20, 30]
            ))
        )
        self.builder.auto_grid(on='point:PPA1')
        getauto = self.builder.get_grid('|*AUTO-point:PPA1*|')
        self.assertEqual(len(df), len(getauto['data']))


if __name__ == '__main__':
    unittest.main()
