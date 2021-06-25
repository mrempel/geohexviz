import unittest

from pandas import DataFrame
from shapely import wkt

from geoviz.utils import geoutils
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, MultiPoint
from testingstructures import TestingShape

testpoints1 = TestingShape(Point(45, 12), Point(60, 12), Point(60, 30),
                           hexids=['8352c6fffffffff', '836221fffffffff', '83430bfffffffff'], condense=False)
testpoly1 = TestingShape(Polygon([[-120.41015624999999, 56.559482483762245], [-112.32421875, 56.559482483762245],
                                  [-112.32421875, 60.326947742998414], [-120.41015624999999, 60.326947742998414],
                                  [-120.41015624999999, 56.559482483762245]]), hexids=
                         ['831208fffffffff', '83122bfffffffff', '831200fffffffff', '831201fffffffff',
                          '831205fffffffff', '831209fffffffff', '831228fffffffff', '83121dfffffffff',
                          '831203fffffffff', '83120dfffffffff', '83121cfffffffff', '83122afffffffff',
                          '83120cfffffffff', '83120efffffffff', '831204fffffffff'])

testline1 = TestingShape(LineString([Point(45, 12), Point(60, 12), Point(60, 30)]),
                         hexids=['8352c6fffffffff', '836221fffffffff', '83430bfffffffff'])

testcollection1 = testpoly1.combine(testpoints1, testpoly1, testline1, testline1.multiply(2, condense=True),
                                    inplace=False)


# TODO: test shapely valid polygons
class TestCases(unittest.TestCase):

    def test_hexify_geodataframe(self):
        # test with Points
        testingdf = GeoDataFrame({'hex': testpoints1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame(geometry=testpoints1.iter_shapes())
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiPoint
        testingdf = GeoDataFrame({'hex': testpoints1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame(
            geometry=testpoints1.condense().iter_shapes())  # condensing multiple Points gives a MultiPoint
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with LineString
        testingdf = GeoDataFrame({'hex': testline1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testline1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiLineString
        testline1.multiply(2, condense=True, inplace=True)  # condensing multiple LineStrings gives a MultiLineString
        testingdf = GeoDataFrame({'hex': testline1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testline1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with Polygon
        testingdf = GeoDataFrame({'hex': testpoly1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testpoly1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiPolygon
        testpoly1.multiply(2, condense=True, inplace=True)  # condensing multiple Polygons gives a MultiPolygon
        testingdf = GeoDataFrame({'hex': testpoly1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testpoly1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with a list of different types
        testingdf = GeoDataFrame({'hex': testcollection1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testcollection1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with a geometry collection
        testcollection1.condense(inplace=True)  # condensing different types gives a GeometryCollection
        testingdf = GeoDataFrame({'hex': testcollection1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testcollection1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(geoutils.hexify_geodataframe(inputdf, 3).sort_index().index))

    def test_bin_by_hex(self):
        """Tests the binning function.
        """

        # binning a single data set will return one for each hex bin
        inputdf = GeoDataFrame({'hex': testpoly1.hexids})
        testingdf = geoutils.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals.pop(), 1)

        # ensuring duplicate hex ids by using duplicate geometries
        inputdf = GeoDataFrame({'hex': testpoly1.multiply(2).hexids})
        testingdf = geoutils.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals.pop(), 2)

        inputdf = GeoDataFrame({'hex': testcollection1.hexids})
        testingdf = geoutils.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertIn(2, vals)
        self.assertIn(4, vals)

    def test_pointify_geodataframe(self):
        testpoly = Polygon([[-120.41015624999999, 56.559482483762245], [-112.32421875, 56.559482483762245],
                            [-112.32421875, 60.326947742998414], [-120.41015624999999, 60.326947742998414],
                            [-120.41015624999999, 56.559482483762245]])
        testpoints = [Point(-120.41015624999999, 56.559482483762245), Point(-112.32421875, 56.559482483762245),
                      Point(-112.32421875, 60.326947742998414), Point(-120.41015624999999, 60.326947742998414),
                      Point(-120.41015624999999, 56.559482483762245)]
        testline = LineString([[-120.41015624999999, 56.559482483762245], [-112.32421875, 56.559482483762245],
                               [-112.32421875, 60.326947742998414], [-120.41015624999999, 60.326947742998414],
                               [-120.41015624999999, 56.559482483762245]])
        testgc = GeometryCollection([*testpoints, testpoly, testline])

        # ensure that pointifying a GeoDataFrame containing a Polygon works
        testgdf = GeoDataFrame(geometry=testpoints, crs='EPSG:4326')

        inputgdf = GeoDataFrame(geometry=[testpoly], crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

        # ensure that pointifying a GeoDataFrame containing a LineString works
        inputgdf = GeoDataFrame(geometry=[testline], crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

        # ensure that pointifying a GeoDataFrame containing a GeometryCollection works
        testgdf = GeoDataFrame(geometry=testpoints * 3, crs='EPSG:4326')

        inputgdf = GeoDataFrame(geometry=[testgc], crs='EPSG:4326')
        resultgdf = geoutils.pointify_geodataframe(inputgdf, keep_geoms=False)
        self.assertTrue(resultgdf.reset_index().geometry.equals(testgdf.reset_index().geometry))

    def test_conform_polygon(self):
        testpoly = wkt.loads(
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))')
        resultpoly = geoutils.conform_polygon(testpoly)

        # all of the Polygon's longitudes should be the same sign if they cross the anti-meridian
        self.assertTrue(all(lon > 0 for lon, _ in resultpoly.exterior.coords))

    def test_conform_geogeometry(self):
        # conform geometry (to avoid 180th meridian issue) and test shapely validity
        testpoly = wkt.loads(
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 '
            '61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))')
        testline = wkt.loads(
            'LINESTRING (169.716796875 61.58549218152362, 183.9111328125 58.19387126497797, 197.314453125 '
            '61.270232790000634, 210.9375 58.309488840677645, 211.3330078125 58.21702494960191)')
        testpoint = wkt.loads('POINT (188.6572265625 61.501734289732326)')

        inputgdf = GeoDataFrame(geometry=[testpoly, testline, testpoint], crs='EPSG:4326')
        resultgdf = geoutils.conform_geogeometry(inputgdf).reset_index()

        # ensure that the first geometry has been changed, as it crosses the anti-meridian
        self.assertEqual(testpoly.geom_type, resultgdf.geometry.values[0].geom_type)
        self.assertNotEqual(testpoly, resultgdf.geometry.values[0])

        # ensure the other geometries remain the same
        self.assertEqual(testline, resultgdf.geometry.values[1])
        self.assertEqual(testpoint, resultgdf.geometry.values[2])

    def test_remove_other_geometries(self):
        testpoly = wkt.loads(
            'POLYGON ((167.51953125 52.214338608258196, -141.767578125 52.214338608258196, -141.767578125 '
            '61.938950426660604, 167.51953125 61.938950426660604, 167.51953125 52.214338608258196))')
        testline = wkt.loads(
            'LINESTRING (169.716796875 61.58549218152362, 183.9111328125 58.19387126497797, 197.314453125 '
            '61.270232790000634, 210.9375 58.309488840677645, 211.3330078125 58.21702494960191)')
        testpoint = wkt.loads('POINT (188.6572265625 61.501734289732326)')

        inputgdf = GeoDataFrame(geometry=[testpoly, testline, testpoint], crs='EPSG:4326')
        resultgdf = geoutils.remove_other_geometries(inputgdf, 'Polygon')
        self.assertTrue(all(t == 'Polygon' for t in resultgdf.geom_type.values))

    def test_check_crossing(self):
        self.assertTrue(geoutils.check_crossing(-178, 181, validate=False))
        self.assertFalse(geoutils.check_crossing(161, 172, validate=False))

    def test_convert_dataframe_coordinates_to_geodataframe(self):
        testpoints = MultiPoint([[10, 10], [20, 20], [30, 30], [40, 40]])
        latitude, longitude = zip(*((p.x, p.y) for p in testpoints))

        testgdf = GeoDataFrame(geometry=[p for p in testpoints], crs='EPSG:4326')

        inputdf = DataFrame(dict(latitude=latitude, longitude=longitude))
        resultgdf = geoutils.convert_dataframe_coordinates_to_geodataframe(inputdf, drop=True, longlat_order=True)

        self.assertTrue(testgdf.equals(resultgdf))

    def test_bin_by_hex2(self):
        testpoints = [Point(10, 10), Point(10, 10), Point(20, 20), Point(20, 20), Point(20, 20), Point(30, 30),
                      Point(45, 45)]

        inputgdf = GeoDataFrame(dict(nums=[{'hello': 3}, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7],
                                     text=['tst1', 'tst2', 'tst3', 'tst4', 'tst5', 'tst6', 'tst7'],
                                     geometry=testpoints), crs='EPSG:4326')

        df = DataFrame(dict(
            lats=[17.57, 17.57, 17.57, 19.98, 19.98, 46.75],
            lons=[10.11, 10.11, 10.12, 50.55, 50.55, 31.17],
            value=[120, 120, 120, 400, 400, 700]
        ))

        df = geoutils.convert_dataframe_coordinates_to_geodataframe(df, latitude_field='lats', longitude_field='lons')
        df = geoutils.hexify_geodataframe(df, hex_resolution=3)
        df = geoutils.bin_by_hex(df, lambda lst: len(lst), add_geoms=True)
        print(df['geometry'])

        resultgdf = geoutils.hexify_geodataframe(inputgdf, add_geoms=True, keep_geoms=False, raise_errors=False)
        print(resultgdf)
        resultgdf2 = geoutils.hexify_geodataframe(GeoDataFrame(geometry=[]), raise_errors=False)
        print(resultgdf)

        import time
        start = time.time()
        tstgdf = geoutils.bin_by_hex_noloss(resultgdf.reset_index(), lambda lst: len(lst), hex_field='hex',
                                            add_geoms=True)
        end = time.time()
        print(f'DIFF {end - start}')

        start = time.time()
        tstgdf = geoutils.bin_by_hex(resultgdf.reset_index(), lambda lst: len(lst), hex_field='hex',
                                     add_geoms=True)
        end = time.time()
        print(f'DIFF {end - start}')

        # print(resultgdf['text'].apply(lambda lst: lst[0] if len(lst)==1 else lst))


if __name__ == '__main__':
    unittest.main()
