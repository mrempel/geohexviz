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
from shapely.geometry import Polygon, Point

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


def alterer(layer):
    layer['data']['value_field'] = -1


class BuilderTestCase(unittest.TestCase):
    """Test cases for the builder module.
    Mainly tests the PlotBuilder class functions.
    """

    def setUp(self):
        self.builder = builder.PlotBuilder()

    def test_read_data_full_full(self):
        """Tests the builder module's ability to read GIS related formats.

        Tests:
        Through the various files included, test reading each of them.
        Also test builtin names.
        """

        # test reading excel file (requires dev-dependencies)
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                dict(
                    path=pjoin(DATA_PATH, "sample-fires2017.xlsx"),
                    sheet_name=0
                )
            )
        )

        # test reading geopackage
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                pjoin(DATA_PATH, "sample-examplegpkg.gpkg")
            )
        )

        # test reading csv
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                pjoin(DATA_PATH, "sample-fires2017.csv")
            )
        )

        # test reading shapefile
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                pjoin(DATA_PATH, "sample-fires2017")
            )
        )

        # test reading kml file
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                pjoin(DATA_PATH, "sample-fires2017.kml")
            )
        )

        # test builtin names
        with self.assertRaises(err.DataReadError):
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                "UNKUNK",
                allow_builtin=True
            )

        # country name
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                "CANADA",
                allow_builtin=True
            )
        )

        # continent name
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                "EUROPE",
                allow_builtin=True
            )
        )

        # test input dict (new)
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                dict(lats=[1,1,1,1,1,1], lons=[2,2,2,2,2,2])
            )
        )

        # test input dict in dict
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                dict(path=dict(lats=[1, 1, 1, 1, 1, 1], lons=[2, 2, 2, 2, 2, 2]))
            )
        )

        # test input str in dict
        self.assertIsNotNone(
            builder._read_data_full(
                "sample",
                err.LayerType.HEXBIN,
                dict(path="CANADA!"),
                allow_builtin=True
            )
        )

    def test_convert_latlong_data(self):
        """Tests the builder module's ability to parse the lat long data from a dataframes.

        Tests:
        Pass various dataframes, geodataframes (empty, coordinate-filled, incorrectly named, etc.) into
        the function and ensure that errors are thrown or the data is read.
        The output data is not tested here.
        """

        # test with empty dataframe
        with self.assertRaises(err.DataEmptyError):
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                DataFrame()
            )

        testdf = DataFrame(dict(
            lats=[1,1,1,1,1,1,1,1,1],
            lons=[2,2,2,2,2,2,2,2,2]
        ))

        # test with coordinate-filled dataframe
        self.assertIsNotNone(
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                testdf
            )
        )

        testdf = DataFrame(dict(
            ynkl=[1,1,1,1,1,1,1,1,1],
            ynkll=[2,2,2,2,2,2,2,2,2]
        ))

        # test with incorrectly-named coordinate-filled dataframe (none specified)
        with self.assertRaises(err.GeometryParseLatLongError):
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                testdf
            )

        # test with incorrectly-named coordinate-filled dataframe (columns specified)
        self.assertIsNotNone(
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                testdf,
                latitude_field="ynkl",
                longitude_field="ynkll"
            )
        )

        # test with empty geodataframe
        with self.assertRaises(err.DataEmptyError):
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                DataFrame()
            )

        testdf = GeoDataFrame(dict(
            ynkl=[1,1,1,1,1,1,1,1,1],
            ynkll=[2,2,2,2,2,2,2,2,2]
        ))

        # test with incorrectly-named coordinate-filled geodataframe (none specified)
        with self.assertRaises(err.GeometryParseLatLongError):
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                testdf
            )

        # test with incorrectly-named coordinate-filled geodataframe (columns specified)
        self.assertIsNotNone(
            builder._convert_latlong_data(
                "sample",
                err.LayerType.HEXBIN,
                testdf,
                latitude_field="ynkl",
                longitude_field="ynkll"
            )
        )


    def test_invalid_naming(self):
        """Tests the builder's ability to detect invalid naming of passed data sets.

        Tests:
        Attempt to add a layer with an invalid name (any non-alphanumeric characters excluding '_' present).
        Ensure that this raises an error.
        Add a layer with a correct name and ensure it was added.
        """
        with self.assertRaises(err.LayerNamingError):
            self.builder.add_outline("+plus+", "CANADA")

        self.builder.add_outline("correct_naming", "CANADA")
        self.assertIsNotNone(self.builder._get_outline("correct_naming"))

    def test_load_input(self):
        """Tests the functions behind loading input data into the correct form.

        Note that this effectively tests the methods used in:
        add_region;
        add_grid;
        add_outline; and
        add_point.

        Tests:
        Input varying valid and invalid dataframes to the builder.
        Ensure correct errors are thrown (if applicable).
        """

        # test empty inputs
        testdf = DataFrame()
        testgdf = GeoDataFrame()
        with self.assertRaises(err.DataEmptyError):
            self.builder.set_hexbin(testdf)
        with self.assertRaises(err.DataEmptyError):
            self.builder.set_hexbin(testgdf)

        # test with missing latitude/longitude entries
        testdf['latitude'] = [1, 2, 3, 4, np.nan, 5]
        testdf['longitude'] = [1, 2, 3, 4, 5, 6]
        self.builder.set_hexbin(testdf)
        self.assertNotEqual(len(testdf), len(self.builder._get_hexbin()['data']))

        # test with no geometry present, and no latitude/longitude columns
        testdf = DataFrame({"value_1": [1, 2, 3, 4, 5], "value_2": [1, 2, 3, 4, 5]})
        self.builder.reset()
        with self.assertRaises(err.GeometryParseLatLongError):
            self.builder.set_hexbin(testdf)
        testgdf = GeoDataFrame({"value_1": [1, 2, 3, 4, 5], "value_2": [1, 2, 3, 4, 5]})
        self.builder.reset()
        with self.assertRaises(err.GeometryParseLatLongError):
            self.builder.set_hexbin(testgdf)

        # test with valid GeoDataFrame input
        testpoints = [Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1)]
        testgdf = GeoDataFrame(geometry=testpoints)
        self.builder.reset()
        self.builder.set_hexbin(testgdf)
        self.assertEqual(len(testpoints), sum(self.builder._get_hexbin()['data']['value_field']))

        # test with invalid latitude/longitude column type
        testdf['latitude'] = [1, 2, 3, 4, 5]
        testdf['longitude'] = [{}, {}, {}, [], {}]
        self.builder.reset()
        with self.assertRaises(err.LatLongParseTypeError):
            self.builder.set_hexbin(testdf)

        # test with empty dataframe but columns present
        testdf = DataFrame(columns=['latitude', 'longitude'])
        self.builder.reset()
        with self.assertRaises(err.DataEmptyError):
            self.builder.set_hexbin(testdf)

    def test_set_hexbin_layer_quantitative(self):
        """Tests quantitative layer functionality.

        Sets the hexbin layer to a quantitative layer,
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

        self.builder.set_hexbin(testdf, hexbin_info=dict(binning_field='val', binning_fn=min))

        gethexbin = self.builder._get_hexbin()  # internal version does not return a deepcopy
        self.assertTrue(gethexbin)  # test to see if the hexbin layer was added correctly
        # ensure the information stored in the layer is valid
        self.assertTrue('data' in gethexbin)
        self.assertTrue('manager' in gethexbin)
        self.assertTrue('value_field' in gethexbin['data'].columns)
        self.assertEqual(gethexbin['DSTYPE'], 'HEX')
        self.assertEqual(gethexbin['VTYPE'], 'NUM')
        # ensure the hexbin dataframe does not reference the dame input dataframe
        self.assertFalse(testdf.equals(gethexbin['data']))
        # ensure the dataframe was binned properly
        self.assertTrue(all(val == 10 for val in gethexbin['data']['value_field']))

    def test_set_hexbin_layer_qualitative(self):
        """Tests the builder's ability to set the hexbin hexbin layer (qualitative).

        Tests:
        Set the hexbin layer with qualitative data.
        Determine if it was stored correctly.
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

        geoms = [wkt.loads(p) for p in testpoints]
        vals = ['EVENDS' if x % 2 == 0 else 'ODDDS' for x in range(len(geoms))]

        testdf = GeoDataFrame(dict(val=vals, geometry=geoms), crs='EPSG:4326')
        self.builder.set_hexbin(
            testdf,
            hexbin_info=dict(
                binning_field='val'
            ),
            manager=dict(
                colorscale=['blue', 'pink']
            )
        )
        gethexbin = self.builder._get_hexbin()  # internal version does not return a deepcopy
        self.assertTrue(gethexbin)  # test to see if the hexbin layer was added correctly
        # ensure the information stored in the layer is valid
        self.assertTrue('data' in gethexbin)
        self.assertTrue('manager' in gethexbin)
        self.assertTrue('value_field' in gethexbin['data'].columns)
        self.assertEqual(gethexbin['DSTYPE'], 'HEX')
        self.assertEqual(gethexbin['VTYPE'], 'STR')
        # ensure the hexbin dataframe does not reference the dame input dataframe
        self.assertFalse(testdf.equals(gethexbin['data']))

    def test_add_region(self):
        """Tests the builder's ability to add region-like layers.

        Tests:
        Add a region-type layer to the builder and determine if it was stored properly.
        """

        self.builder.add_region('RRA1', 'CANADA')
        getreg = self.builder._get_region('RRA1')
        self.assertEqual(getreg['DSTYPE'], 'RGN')
        self.assertTrue('value_field' in getreg['data'])
        self.assertTrue(all(v == 0 for v in getreg['data']['value_field']))

        self.builder.add_region('RRA2', pjoin(DATA_PATH, 'sample-canbuffer'))
        getreg = self.builder._get_region('RRA2')
        self.assertEqual(getreg['DSTYPE'], 'RGN')
        self.assertTrue('value_field' in getreg['data'])
        self.assertTrue(all(v == 0 for v in getreg['data']['value_field']))

    def test_add_grid(self):
        """Tests the builder's ability to add grid-like layers.

        Tests:
        Add a grid-type layer to the builder and determine if it was stored properly.
        """

        self.builder.add_grid('GGA1', 'FRANCE')
        getgr = self.builder._get_grid('GGA1')
        self.assertEqual(getgr['DSTYPE'], 'GRD')
        self.assertTrue('value_field' in getgr['data'])
        self.assertTrue(all(v == 0 for v in getgr['data']['value_field']))

        self.builder.add_grid('GGA2', pjoin(DATA_PATH, 'sample-canbuffer'))
        getgr = self.builder._get_grid('GGA2')
        self.assertEqual(getgr['DSTYPE'], 'GRD')
        self.assertTrue('value_field' in getgr['data'])
        self.assertTrue(all(v == 0 for v in getgr['data']['value_field']))

    def test_add_outline(self):
        """Tests the builder's ability to add outline-like layers.

        Tests:
        Add a outline-type layer to the builder and determine if it was stored properly.
        """
        self.builder.add_outline('OOA1', 'SOUTH AMERICA')
        getout = self.builder._get_outline('OOA1')
        self.assertEqual(getout['DSTYPE'], 'OUT')
        self.assertTrue('value_field' in getout['data'])
        self.assertTrue(all(v == 0 for v in getout['data']['value_field']))

        self.builder.add_outline('OOA2', pjoin(DATA_PATH, 'sample-canbuffer'))
        getout = self.builder._get_outline('OOA2')
        self.assertEqual(getout['DSTYPE'], 'OUT')
        self.assertTrue('value_field' in getout['data'])
        self.assertTrue(all(v == 0 for v in getout['data']['value_field']))

    def test_add_points(self):
        """Tests the builder's ability to add point-like layers.

        Tests:
        Add a point-type layer to the builder and determine if it was stored properly.
        """

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
        """Test the builder's ability to clip layers to other layers.

        Lengthier runtime.
        Tests:
        Clip layers within the builder to each other. Ensure the resulting
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

        self.builder.set_hexbin(GeoDataFrame(geometry=testdata, crs='EPSG:4326'))

        # add regions for Germany and Poland
        self.builder.add_region('RRA1', 'GERMANY')
        self.builder.add_region('RRA2', 'POLAND')

        gethexbin = self.builder._get_hexbin()
        initiallen = len(gethexbin['data'])

        # test clipping hexbin to the first region alone
        self.builder.clip_layers('hexbin', 'region:RRA1')
        newlen = len(self.builder._get_hexbin()['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(5, newlen)  # for this layer

        # test clipping hexbin to the second region alone
        self.builder.reset_hexbin_data()  # reset back to original for another clip
        self.builder.clip_layers('hexbin', 'region:RRA2')
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(5, newlen)  # for this layer

        # test clipping hexbin to all regions
        self.builder.reset_hexbin_data()
        self.builder.clip_layers('hexbin', 'regions')
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(10, newlen)  # for this layer

        # add a grid over europe
        self.builder.add_grid('GGA1', 'EUROPE')
        # add an outline over france and italy
        self.builder.add_outline('OOA1', 'FRANCE')
        self.builder.add_outline('OOA2', 'ITALY')

        # test clipping grids to hexbin
        self.builder.reset_hexbin_data()
        gridlen = sum(self.builder.apply_to_query('grids', lambda layer: len(layer['data'])))
        self.builder.clip_layers('grids', 'hexbin', operation='within')
        newgridlen = sum(self.builder.apply_to_query('grids', lambda layer: len(layer['data'])))
        self.assertLess(newgridlen, gridlen)  # for this layer
        self.assertEqual(14, newgridlen)  # for this layer

        # test clipping hexbin to grids
        self.builder.reset_hexbin_data()
        self.builder.clip_layers('hexbin', 'grids', operation='within')  # intersects will get a different result
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(14, newlen)  # for this layer

        # test clipping hexbin to the first outline alone
        self.builder.reset_hexbin_data()
        self.builder.clip_layers('hexbin', 'outline:OOA1')
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(2, newlen)  # for this layer

        # test clipping hexbin to the second outline alone
        self.builder.reset_hexbin_data()
        self.builder.clip_layers('hexbin', 'outline:OOA2')
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(1, newlen)  # for this layer

        # test clipping hexbin to all outlines
        self.builder.reset_hexbin_data()
        self.builder.clip_layers('hexbin', 'outlines', method='gpd')
        newlen = len(gethexbin['data'])
        self.assertLess(newlen, initiallen)  # for this layer
        self.assertEqual(3, newlen)  # for this layer

        self.builder.reset_hexbin_data()
        outlen = sum(self.builder.apply_to_query('outlines', lambda layer: len(layer['data'])))
        self.builder.clip_layers('outlines', 'hexbin', operation='intersects')
        newoutlen = sum(self.builder.apply_to_query('outlines', lambda layer: len(layer['data'])))
        self.assertLessEqual(newoutlen, outlen)
        self.assertEqual(2, newoutlen)  # for this layer

        # test clipping hexbin to points, which does not make sense
        self.builder.reset_hexbin_data()
        self.builder.add_point('PPA1', GeoDataFrame(geometry=testdata, crs='EPSG:4326'))

        err = False
        try:
            self.builder.clip_layers('hexbin', 'points', method='gpd')
        except ValueError:
            err = True
        self.assertTrue(err)

    def test_logify_scale(self):
        """Tests the builder's ability to make a logarithmic scale.

        This test also tests most of the logify_info() functionality (in plot_util).

        Tests:
        Set the hexbin layer and invoke the function. Ensure that the layer's
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

        self.builder.set_hexbin(GeoDataFrame(dict(geometry=testdata, val=values), crs='EPSG:4326'),
                                hexbin_info=dict(binning_fn=min, binning_field='val'))

        gethexbin = self.builder._get_hexbin()
        oldvals = gethexbin['data']['value_field']
        self.builder.logify_scale()
        newvals = gethexbin['data']['value_field']
        nvalmax, nvalmin = max(newvals), min(newvals)
        start, end = int(math.floor(nvalmin)), int(math.floor(nvalmax)) + 1

        # ensure the values are logged
        self.assertTrue(all(y == np.log10(x) for x, y in zip(oldvals, newvals)))

        # ensure all necessary values are in colorbar properties
        self.assertTrue(all(x in gethexbin['manager']['colorbar']['tickvals'] for x in range(start, end)))

    def test_adjust_opacity(self):
        """Tests the adjusting opacity feature of the builder.

        Tests:
        Set the hexbin layer with a custom colorscale and invoke the function.
        Ensure the opacity is present within the colors of the colorscale.
        """
        with self.assertRaises(err.NoLayerError):
            self.builder.adjust_opacity()

        inp_colorscale = [[0, 'rgb(10, 10, 10)'], [0.5, 'rgb(50, 50, 50)'], [1, 'rgb(90, 90, 90)']]
        inp_opacity = 0.6
        self.builder.set_hexbin(
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
        out_colorscale = self.builder._get_hexbin()['manager']['colorscale']
        self.assertTrue(all(str(inp_opacity) in ci for _, ci in out_colorscale))

    def test_discretize_scale(self):
        """Tests the builder's ability to make a discrete scale.

        Tests:
        Set the hexbin layer alongside a custom colorscale and invoke the function.
        Ensure that the colors are present twice in the output colorscale.
        """
        with self.assertRaises(err.NoLayerError):
            self.builder.discretize_scale()

        inp_colorscale = [[0, 'rgb(10, 10, 10)'], [0.5, 'rgb(50, 50, 50)'], [1, 'rgb(90, 90, 90)']]
        # inp_colorscale = ['red', 'blue', 'green']
        inp_ds = 1
        self.builder.set_hexbin(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )),
            manager=dict(
                zmin=1,
                colorscale=inp_colorscale
            )
        )
        gethexbin = self.builder._get_hexbin()
        self.builder.discretize_scale(discrete_size=inp_ds)
        out_colorscale = gethexbin['manager']['colorscale']
        inpcolors = [ci for _, ci in inp_colorscale]
        outcolors = [ci for _, ci in out_colorscale]
        self.assertTrue(all(outcolors.count(ci) == 2 for ci in inpcolors))

    def test_adjust_focus(self):
        """Tests the builders ability to adjust the focus of the plot.

        Tests:
        Add a layer to the builder and focus onto it.
        Ensure that the geospatial properties have been set properly.
        """
        df = DataFrame(dict(
            latitude=[10, 10, 10, 10, 20, 20],
            longitude=[20, 20, 10, 10, 10, 10]
        ))
        self.builder.add_point('PPA1', df)
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
        Add layers and apply a query to them. Ensure the resulting
        layers have been altered in the correct way.
        """
        self.builder.set_hexbin(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        gethexbin = self.builder._get_hexbin()
        self.builder.apply_to_query('all', alterer)
        self.assertTrue(all(x == -1 for x in gethexbin['data']['value_field'].values))

    def test_update_hexbin_manager(self):
        """Tests the builder's ability to update the hexbin layer's manager.

        Tests:
        Set the hexbin layer and update it. Check if it was updated properly.
        """
        with self.assertRaises(err.NoLayerError):
            self.builder.update_hexbin_manager(colorscale='Viridis')
        self.builder.set_hexbin(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        self.builder.update_hexbin_manager(colorscale='Picnic')
        self.assertEqual('Picnic', self.builder._get_hexbin()['manager']['colorscale'])

    def test_clear_hexbin_manager(self):
        """Tests the builder's ability to clear the manager of the hexbin layer.

        Tests:
        Set the hexbin layer and clear it's manager. Ensure the manager is empty.
        """
        with self.assertRaises(err.NoLayerError):
            self.builder.clear_hexbin_manager()
        self.builder.set_hexbin(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))
        self.builder.clear_hexbin_manager()
        self.assertEqual({}, self.builder._get_hexbin()['manager'])

    def test_reset_hexbin_data(self):
        """Tests the builder's ability to reset the hexbin layer's data back to its original state.

        Tests:
        Set the hexbin layer. Alter it's data and reset it. Ensure that the data is equal to the original data.
        """
        with self.assertRaises(err.NoLayerError):
            self.builder.reset_hexbin_data()

        self.builder.set_hexbin(
            DataFrame(dict(
                latitude=[10, 10, 10, 10, 20, 20],
                longitude=[20, 20, 10, 10, 10, 10]
            )))

        gethexbin = self.builder._get_hexbin()
        vals = gethexbin['data']['value_field'].copy()
        self.builder.apply_to_query('hexbin', alterer)
        self.assertTrue(all(x == -1 for x in gethexbin['data']['value_field'].values))
        self.builder.reset_hexbin_data()
        self.assertTrue(gethexbin['data']['value_field'].equals(vals))

    def test_update_region_manager(self):
        """Tests the builder's ability to update the manager of region layers.

        Tests:
        Add region layers and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.NoLayerError):
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
        """Tests the builder's ability to clear the manager of region layers.

        Tests:
        Add region layers and clear their managers. Ensure they are empty.
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
        """Tests the builder's ability to reset the data of region layers.

        Tests:
        Add region layers and alter their data. Revert the data. Ensure they are back to original state.
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
        """Tests the builder's ability to update the manager of grid layers.

        Tests:
        Add grid layers and update their managers. Ensure they were updated correctly.
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
        """Tests the builder's ability to clear the manager of grid layers.

        Tests:
        Add grid layers and clear their managers. Ensure they are empty.
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
        """Tests the builder's ability to reset the data of grid layers.

        Tests:
        Add grid layers and alter their data. Revert the data. Ensure they are back to original state.
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
        """Tests the builder's ability to update the manager of outline layers.

        Tests:
        Add outline layers and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.NoLayerError):
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
        """Tests the builder's ability to clear the manager of outline layers.

        Tests:
        Add outline layers and clear their managers. Ensure they are empty.
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
        """Tests the builder's ability to reset the data of outline layers.

        Tests:
        Add outline layers and alter their data. Revert the data. Ensure they are back to original state.
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
        """Tests the builder's ability to update the manager of point layers.

        Tests:
        Add point layers and update their managers. Ensure they were updated correctly.
        """
        with self.assertRaises(err.NoLayerError):
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
        """Tests the builder's ability to clear the manager of point layers.

        Tests:
        Add point layers and clear their managers. Ensure they are empty.
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
        """Tests the builder's ability to reset the data of point layers.

        Tests:
        Add point layers and alter their data. Revert the data. Ensure they are back to original state.
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
        df = DataFrame(dict(
                latitude=[10, 20, 30],
                longitude=[10, 20, 30]
            ))
        self.builder.add_point('PPA1', df)
        self.builder.auto_grid(on='point:PPA1')
        getauto = self.builder.get_grid('|*AUTO-point:PPA1*|')
        self.assertEqual(len(df), len(getauto['data']))


if __name__ == '__main__':
    unittest.main()
