import unittest
from typing import List
from pandas import DataFrame
import geopandas as gpd
from geohexviz.utils import geoutils
from geopandas import GeoDataFrame
from shapely.geometry import GeometryCollection, MultiPoint, Point, MultiLineString, LineString, Polygon
from shapely import wkt


def shapes_from_wkt(*args) -> List[str]:
    """Reads a many geometries from wkt format.

    :param args: The shapes in wkt format
    :type args: *args
    :return: The read shapes in shapely format
    :rtype: List[str]
    """
    return [wkt.loads(s) for s in args]


# TODO: test shapely valid polygons
class GeoUtilsTestCase(unittest.TestCase):
    """Test class for tests concerning the geoutils module.
    """

    def test_hexify_geodataframe(self):
        """Tests the functionality behind generating a hexified geodataframe.

        Tests:
        Having sample hexids and their corresponding geometries, attempt to generate a geodataframe
        with equal hex ids and geometries. Ensure that the hex ids are the expected hex ids. Also
        ensure that old geometry is present if specified.
        """

        testpoints = shapes_from_wkt(*[
            'POINT (-33.75 56.75272287205736)',
            'POINT (43.9453125 64.77412531292873)',
            'POINT (33.046875 48.922499263758255)',
            'POINT (72.421875 47.517200697839414)'
        ])
        testpoints_hexids = [
            '831b62fffffffff',
            '83115efffffffff',
            '831e6afffffffff',
            '83202afffffffff'
        ]

        testmultipoint = shapes_from_wkt(*[
            'MULTIPOINT (-30.234375 47.040182144806664, -9.84375 60.930432202923335, 4.921875 52.908902047770255, '
            '-37.265625 36.31512514748051, -7.3828125 42.032974332441405) '
        ])
        testmultipoint_hexids = [
            '831a68fffffffff',
            '831931fffffffff',
            '831968fffffffff',
            '83359efffffffff',
            '833923fffffffff'
        ]

        testlinestring = shapes_from_wkt(*[
            'LINESTRING (-72.421875 12.554563528593656, -36.9140625 16.29905101458183, -11.25 31.952162238024975, '
            '-18.28125 8.059229627200192, 8.7890625 -6.315298538330033, 32.34375 9.795677582829743) '
        ])
        testlinestring_hexids = [
            '83670bfffffffff',
            '835633fffffffff',
            '8339b1fffffffff',
            '837cd2fffffffff',
            '83822efffffffff',
            '836a66fffffffff'
        ]

        testmultilinestring = shapes_from_wkt(*[
            'MULTILINESTRING ((4.921875 36.87962060502676, 43.2421875 43.068887774169625, 69.60937499999999 '
            '37.996162679728116, -13.0078125 21.289374355860424, 28.828124999999996 27.059125784374068, 50.2734375 '
            '2.1088986592431382, 68.203125 16.97274101999902))'
        ])
        testmultilinestring_hexids = [
            '83395bfffffffff',
            '832c2cfffffffff',
            '832084fffffffff',
            '835518fffffffff',
            '833e54fffffffff',
            '837aa4fffffffff',
            '8360ddfffffffff'
        ]

        testpolygon = shapes_from_wkt(*[
            'POLYGON ((63.21533203124999 39.5633531658293, 61.25976562499999 38.865374851611634, 62.65502929687499 '
            '37.37888785004527, 63.599853515625 35.84453450421662, 65.41259765625 37.39634613318923, 65.753173828125 '
            '39.774769485295465, 62.35839843749999 42.13082130188811, 59.78759765625 41.64007838467894, '
            '63.21533203124999 39.5633531658293)) '
        ])
        testpolygon_hexids = [
            '834368fffffffff',
            '832198fffffffff',
            '832183fffffffff',
            '83219efffffffff',
            '832182fffffffff',
            '83219afffffffff',
            '83436dfffffffff',
            '832193fffffffff',
            '834369fffffffff',
            '83219cfffffffff',
            '832180fffffffff'
        ]

        testmultipolygon = shapes_from_wkt(*[
            'MULTIPOLYGON (((61.72119140625 40.01078714046552, 60.765380859375 39.65645604812829, 60.34790039062501 '
            '38.71980474264237, 62.32543945312499 38.20365531807149, 64.51171875 39.198205348894795, '
            '62.95166015624999 40.65563874006118, 61.72119140625 40.01078714046552), (62.32543945312499 '
            '36.94111143010769, 60.040283203125 35.33529320309328, 63.4130859375 34.361576287484176, 64.962158203125 '
            '35.808904044068626, 62.91870117187499 37.22158045838649, 62.32543945312499 36.94111143010769))) '
        ])
        testmultipolygon_hexids = [
            '83219efffffffff',
            '832191fffffffff',
            '832193fffffffff',
            '83219cfffffffff',
            '832190fffffffff'
        ]

        testlinearring = shapes_from_wkt(*[
            'LINEARRING (63.21533203124999 39.5633531658293, 61.25976562499999 38.86537485161163, 62.65502929687499 '
            '37.37888785004527, 63.599853515625 35.84453450421662, 65.41259765625 37.39634613318923, 65.753173828125 '
            '39.77476948529547, 62.35839843749999 42.13082130188811, 59.78759765625 41.64007838467894, '
            '63.21533203124999 39.5633531658293) '
        ])
        testlinearring_hexids = [
            '83219efffffffff',
            '832190fffffffff',
            '83436cfffffffff',
            '83436afffffffff',
            '834369fffffffff',
            '832199fffffffff',
            '832180fffffffff',
            '8321b1fffffffff',
            '83219efffffffff'
        ]

        testgeometrycollection = [GeometryCollection(shapes_from_wkt(*[
            'POINT (62.99560546875 31.3348710339506)',
            'POINT (64.7314453125 31.015278981711266)',
            'POINT (63.2373046875 30.486550842588485)',
            'LINESTRING (63.5009765625 34.687427949314845, 66.20361328125 35.746512259918504, 68.90625 '
            '35.17380831799959, 69.80712890625 34.21634468843463)',
            'MULTIPOLYGON (((66.11572265625 34.415973384481866, 64.4677734375 33.96158628979907, 64.84130859375 '
            '32.32427558887655, 67.17041015625 32.32427558887655, 68.5107421875 33.15594830078649, 67.5439453125 '
            '34.488447837809304, 66.11572265625 34.415973384481866), (67.7197265625 31.55981453201843, 67.0166015625 '
            '30.12612436422458, 68.92822265625 29.897805610155874, 70.09277343749999 30.543338954230222, 69.873046875 '
            '31.240985378021307, 67.7197265625 31.55981453201843))) '
        ]))]
        testgeometrycollection_hexids = [
            '834351fffffffff',
            '83435afffffffff',
            '834353fffffffff',
            '834341fffffffff',
            '83434dfffffffff',
            '832094fffffffff',
            '832092fffffffff',
            '83434afffffffff',
            '83434efffffffff',
            '83434bfffffffff',
            '834264fffffffff',
            '834265fffffffff',
            '83435dfffffffff'
        ]

        # test empty dataframe
        with self.assertRaises(ValueError):
            geoutils.hexify_dataframe(GeoDataFrame(geometry=[], dtype='object'), 3, raise_errors=True)
        with self.assertRaises(ValueError):
            geoutils.hexify_dataframe(GeoDataFrame(dtype='object'), 3, raise_errors=True)

        resultdf = geoutils.hexify_dataframe(GeoDataFrame(), 2, raise_errors=False)
        self.assertTrue(resultdf.empty)

        # test with Points
        inputdf = GeoDataFrame(geometry=testpoints, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS',
                                             as_index=False)
        self.assertIn('OLDGEOMS', list(resultdf.columns.values))
        self.assertTrue(
            all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))  # check if the old geometry is still present
        self.assertListEqual(list(sorted(testpoints_hexids)), list(sorted(resultdf['HEX'].values)))

        # test with MultiPoint
        inputdf = GeoDataFrame(geometry=testmultipoint, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=False, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf.geometry for g in inputdf.geometry))  # old geometry is still present
        self.assertListEqual(list(sorted(testmultipoint_hexids)), list(resultdf.sort_index().index.values))

        # test with LineString
        inputdf = GeoDataFrame(geometry=testlinestring, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testlinestring_hexids)), list(resultdf.sort_index().index.values))

        # test with MultiLineString
        inputdf = GeoDataFrame(geometry=testmultilinestring, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testmultilinestring_hexids)), list(resultdf.sort_index().index.values))

        # test with LinearRing
        inputdf = GeoDataFrame(geometry=testlinearring, crs='EPSG:4326', dtype='object')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testlinearring_hexids)), list(resultdf.sort_index().index.values))

        # test with Polygon
        inputdf = GeoDataFrame(geometry=testpolygon, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testpolygon_hexids)), list(resultdf.sort_index().index.values))

        # test with MultiPolygon
        inputdf = GeoDataFrame(geometry=testmultipolygon, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testmultipolygon_hexids)), list(resultdf.sort_index().index.values))

        # test with a list of different types
        inputdf = GeoDataFrame(geometry=[*testpolygon, *testlinestring, *testmultipolygon], crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted([*testpolygon_hexids, *testlinestring_hexids, *testmultipolygon_hexids])),
                             list(resultdf.sort_index().index.values))

        # test with a geometry collection
        inputdf = GeoDataFrame(geometry=testgeometrycollection, crs='EPSG:4326')
        resultdf = geoutils.hexify_dataframe(inputdf, 3, add_geom=True, keep_geom=True, old_geom_name='OLDGEOMS')
        self.assertTrue(all(g in resultdf['OLDGEOMS'] for g in inputdf.geometry))
        self.assertListEqual(list(sorted(testgeometrycollection_hexids)), list(resultdf.sort_index().index.values))

    def test_hexify_geometry(self):
        """This tests adding hex cell geometry to a dataframe.

        Tests:
        Take a dataframe containing hex ids and add their corresponding geometries.
        Check if the geometries are the expected geometries.
        """

        hid_geoms = shapes_from_wkt(*[
            'POLYGON ((61.80927954354407 31.60447526399805, 62.23463004844746 31.06969011324834, 62.99260769721348 '
            '31.09020285137916, 63.33771104629135 31.64758171007723, 62.91614862847988 32.1894752389768, '
            '62.14552596214781 32.16685264323142, 61.80927954354407 31.60447526399805))',
            'POLYGON ((64.16169773787351 30.56586289378155, 64.56432678445968 30.0264039738691, 65.31584973168054 '
            '30.03224494110971, 65.67707410461038 30.57887942947722, 65.27914927306159 31.12518155265482, '
            '64.51512798905058 31.11798740295368, 64.16169773787351 30.56586289378155))',
            'POLYGON ((61.90309230297799 30.51386893491829, 62.32083136938339 29.9835514419504, 63.06656926975328 '
            '30.00202145311769, 63.40664700727329 30.55282352546525, 62.99260769721348 31.09020285137916, '
            '62.23463004844746 31.06969011324834, 61.90309230297799 30.51386893491829))',
            'POLYGON ((62.75521552163305 34.41862175518751, 63.19266565275657 33.86881521203879, 63.98677550291011 '
            '33.88720955394923, 64.35716786356136 34.45727863719613, 63.92434226966045 35.01417583784234, '
            '63.11630350339601 34.99388776933781, 62.75521552163305 34.41862175518751))',
            'POLYGON ((65.11920872149379 35.60259272415173, 65.54672898479255 35.03871987385368, 66.36044000601737 '
            '35.04289546846444, 66.76110814038532 35.6120700580808, 66.3396008040714 36.18277956307367, '
            '65.51119773173042 36.17746302027599, 65.11920872149379 35.60259272415173)) '
        ])

        # test with Points
        inputdf = DataFrame(dict(ids=[
            '834351fffffffff',
            '83435afffffffff',
            '834353fffffffff',
            '834341fffffffff',
            '83434dfffffffff'
        ]))

        testdf = GeoDataFrame(geometry=hid_geoms, crs='EPSG:4326')
        resultdf = geoutils.hexify_geometry(inputdf, hex_col='ids')
        # some numbers generated may not be EXACTLY the same
        self.assertTrue(all(resultdf.geom_equals_exact(testdf.geometry, tolerance=0.000000001).values))

    def test_pointify_geodataframe(self):
        """Tests the module's ability to convert a geodataframe into one containing corresponding point boundaries.

        Tests:
        Take a sample dataframe containing different shapes and convert it. Check if the
        newly generated geometries represent the boundary of the shapes they came from.
        """

        testpoly = shapes_from_wkt(*[
            'POLYGON ((-120.41015625 56.55948248376225, -112.32421875 56.55948248376225, '
            '-112.32421875 60.32694774299841, -120.41015625 60.32694774299841, -120.41015625 '
            '56.55948248376225))'
        ])

        testpoints = shapes_from_wkt(*[
            'POINT (-120.41015625 56.55948248376225)',
            'POINT (-112.32421875 56.55948248376225)',
            'POINT (-112.32421875 60.32694774299841)',
            'POINT (-120.41015625 60.32694774299841)',
            'POINT (-120.41015625 56.55948248376225)'
        ])

        testline = shapes_from_wkt(*[
            'LINESTRING (-120.41015625 56.55948248376225, -112.32421875 56.55948248376225, -112.32421875 '
            '60.32694774299841, -120.41015625 60.32694774299841, -120.41015625 56.55948248376225) '
        ])

        testgc = [GeometryCollection([*testpoints, *testpoly, *testline])]

        # ensure that pointifying a GeoDataFrame containing a Polygon works
        testgdf = GeoDataFrame(geometry=testpoints, crs='EPSG:4326')
        inputgdf = GeoDataFrame(geometry=testpoly, crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

        # ensure that pointifying a GeoDataFrame containing a LineString works
        inputgdf = GeoDataFrame(geometry=testline, crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

        # ensure that pointifying a GeoDataFrame containing a GeometryCollection works
        testgdf = GeoDataFrame(geometry=testpoints * 3, crs='EPSG:4326')
        inputgdf = GeoDataFrame(geometry=testgc, crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

    def test_conform_polygon(self):
        """Test the module's ability to conform a Polygon to a geo standard.

        Tests:
        Take a sample Polygon that crosses the anti-meridian, and conform it.
        """
        testpoly = shapes_from_wkt(
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 '
            '61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))')[0]
        resultpoly = geoutils.conform_polygon(testpoly)

        # all of the Polygon's longitudes should be the same sign if they cross the anti-meridian
        self.assertTrue(all(lon > 0 for lon, _ in resultpoly.exterior.coords))

    def test_conform_geogeometry(self):
        """Test the module's ability to conform a geodataframe's geometry to a standard.

        Tests:
        Take a geodataframe and conform it. Ensure that the resulting geometry only has the Polygon-like
        geometry altered.
        """

        # conform geometry (to avoid 180th meridian issue) and test shapely validity
        testshapes = shapes_from_wkt(*[
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 '
            '61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))',
            'LINESTRING (169.716796875 61.58549218152362, 183.9111328125 58.19387126497797, 197.314453125 '
            '61.270232790000634, 210.9375 58.309488840677645, 211.3330078125 58.21702494960191)',
            'POINT (188.6572265625 61.501734289732326)'])

        inputgdf = GeoDataFrame(geometry=testshapes, crs='EPSG:4326')
        resultgdf = geoutils.conform_geogeometry(inputgdf).reset_index()

        # ensure that the first geometry has been changed, as it crosses the anti-meridian
        self.assertEqual(testshapes[0].geom_type, resultgdf.geometry.values[0].geom_type)
        self.assertNotEqual(testshapes[0], resultgdf.geometry.values[0])

        # ensure the other geometries remain the same
        self.assertEqual(testshapes[1], resultgdf.geometry.values[1])
        self.assertEqual(testshapes[2], resultgdf.geometry.values[2])

    def test_remove_other_geometries(self):
        """Test the module's ability to remove geometry types from a geodataframe.

        Tests:
        Take a geodataframe with varying types of geometries and remove any geometries that aren't Polygons.
        Ensure that the resulting geometry types in the resulting geodataframe are only Polygons.
        """

        inputgdf = GeoDataFrame(geometry=shapes_from_wkt(*[
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 '
            '61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))',
            'LINESTRING (169.716796875 61.58549218152362, 183.9111328125 58.19387126497797, 197.314453125 '
            '61.270232790000634, 210.9375 58.309488840677645, 211.3330078125 58.21702494960191)',
            'POINT (188.6572265625 61.501734289732326)'
        ]), crs='EPSG:4326')
        resultgdf = geoutils.remove_other_geometries(inputgdf, 'Polygon')
        self.assertTrue(all(t == 'Polygon' for t in resultgdf.geom_type.values))

    def test_check_crossing(self):
        """Tests the module's ability to check if a line connecting two points crosses the anti-meridian.

        Tests:
        Take two points that certainly cross the anti-meridian, and ensure that they do.
        """
        self.assertTrue(geoutils.check_crossing(-178, 181, validate=False))
        self.assertFalse(geoutils.check_crossing(161, 172, validate=False))

    def test_bin_by_hex(self):
        """Test the module's ability to bin a dataframe by hexagonal id.

        Tests:
        Take a dataframe containing multiple duplicate hex ids. Bin the dataframe and
        ensure that the resulting numerical column corresponds to the instances of each
        entry are the same amount as the duplicates within the original dataframe.
        """

        testingdf = DataFrame(dict(
            ids=[
                    '834351fffffffff',
                    '83435afffffffff',
                    '834353fffffffff',
                    '834341fffffffff',
                    '83434dfffffffff'
                ] * 3
        ))

        resultdf = geoutils.bin_by_hexid(testingdf, hex_field='ids', add_geoms=True)
        self.assertTrue(all(v == 3 for v in resultdf['value_field']))

    def test_get_present_geometries(self):
        """Tests the module's ability to obtain all of the unique geometry types within a geodataframe.

        Tests:
        Take a geodataframe with multipart geometries and invoke the function.
        Ensure the resulting set of unique geometry types is correct.
        """
        testdf = GeoDataFrame(geometry=[
            GeometryCollection([Point(0, 0), Point(1, 1),
                                GeometryCollection([MultiLineString([[[1, 1], [0, 0]], [[10, 10], [20, 20]]])])]),
            LineString([[1, 1], [0, 0]])
        ])
        self.assertListEqual(list(sorted(['Point', 'MultiLineString', 'LineString'])),
                             list(sorted(
                                 geoutils.get_present_geomtypes(testdf, allow_collections=True, collapse_geoms=False))))
        self.assertListEqual(list(sorted(['Point', 'LineString'])),
                             list(sorted(
                                 geoutils.get_present_geomtypes(testdf, allow_collections=True, collapse_geoms=True))))

    def test_generate_grid_over(self):
        """Tests the module's ability to generate a grid over a dataframe (bbox).

        Tests:
        Take two dataframes. One with multiple geometries and one with a single geometry.
        Ensure that the all of the resulting hexes fall intersect with the bbox.
        The single geometry dataframe should return an empty dataframe.
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
        inputdf = GeoDataFrame(geometry=testdata, crs='EPSG:4326')
        testdf = GeoDataFrame(geometry=[Polygon.from_bounds(*inputdf.total_bounds)], crs='EPSG:4326')
        resultdf = geoutils.generate_grid_over(inputdf, 3)
        resulting = gpd.sjoin(resultdf, testdf, op='intersects')
        self.assertEqual(len(resultdf), len(resulting))
        self.assertTrue(all(i == 0 for i in resulting['index_right']))

        # edge case: only one geometry
        testdata = shapes_from_wkt('POINT (3.9111328125000004 48.45835188280866)')
        inputdf = GeoDataFrame(geometry=testdata, crs='EPSG:4326')
        testdf = GeoDataFrame(geometry=[Polygon.from_bounds(*inputdf.total_bounds)], crs='EPSG:4326')
        resultdf = geoutils.generate_grid_over(inputdf, 3)
        resulting = gpd.sjoin(resultdf, testdf, op='intersects')
        self.assertEqual(len(resultdf), len(resulting))
        self.assertTrue(all(i == 0 for i in resulting['index_right']))


if __name__ == '__main__':
    unittest.main()
