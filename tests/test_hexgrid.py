import unittest
from geoviz import hexgrid
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint, GeometryCollection
from pandas import Series


class TestingShape:
    multitype = {
        Point: MultiPoint,
        Polygon: MultiPolygon,
        LineString: MultiLineString
    }

    def __init__(self, *args, hexids=None, multiply=False, condense=False):
        if hexids is None:
            hexids = []
        if len(args) > 1:
            self.shapes = list(args)
        else:
            self.shapes = args[0]

        self.hexids = hexids

        if multiply:
            self.multiply(multiply, condense=False, inplace=True)

        if condense:
            self.condense(inplace=True)

    def __str__(self):
        return str({
            'shapes': self.shapes,
            'hexids': self.hexids
        })

    def _transfer(self, tst):
        self.shapes = tst.shapes
        self.hexids = tst.hexids

    def iter_shapes(self):
        if isinstance(self.shapes, list):
            return self.shapes.copy()
        else:
            return [self.shapes]

    def condense(self, inplace=False):

        notfound = True
        for typer in self.multitype:

            shapes = self.iter_shapes()
            if all(isinstance(x, typer) for x in shapes):
                newtst = TestingShape(self.multitype[typer](shapes), hexids=self.hexids)
                notfound = False
                break

        if notfound:
            newtst = TestingShape(GeometryCollection(self.iter_shapes()), hexids=self.hexids)

        if inplace:
            self._transfer(newtst)
        else:
            return newtst

    def decondense(self, inplace=False):

        newtst = TestingShape(*(x for x in self._get_shapes()), hexids=self.hexids)
        if inplace:
            self._transfer(newtst)
        else:
            return newtst

    def multiply(self, num, condense=False, inplace=False):

        newtst = TestingShape(*(self.iter_shapes() * num), hexids=self.hexids * num, condense=condense)
        if inplace:
            self._transfer(newtst)
        else:
            return newtst

    def __copy__(self):
        return TestingShape(*self.iter_shapes(), hexids=self.hexids.copy(), condense=False)

    def combine(self, *args, condense=False, inplace=False):

        newtst = self.__copy__()

        def helper(other):
            newtst.shapes = newtst.iter_shapes()
            newtst.shapes.extend(other.iter_shapes())
            newtst.hexids = list(newtst.hexids)
            newtst.hexids.extend(other.hexids)
            if condense:
                newtst.condense(inplace=True)
                return newtst

        for item in args:
            helper(item)

        if inplace:
            self._transfer(newtst)
        else:
            return newtst


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


class TestCases(unittest.TestCase):

    def test_grid_id_generation(self):
        # test with Points
        testingdf = GeoDataFrame({'hex': testpoints1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame(geometry=testpoints1.iter_shapes())
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiPoint
        testingdf = GeoDataFrame({'hex': testpoints1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame(
            geometry=testpoints1.condense().iter_shapes())  # condensing multiple Points gives a MultiPoint
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with LineString
        testingdf = GeoDataFrame({'hex': testline1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testline1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiLineString
        testline1.multiply(2, condense=True, inplace=True)  # condensing multiple LineStrings gives a MultiLineString
        testingdf = GeoDataFrame({'hex': testline1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testline1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with Polygon
        testingdf = GeoDataFrame({'hex': testpoly1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testpoly1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with MultiPolygon
        testpoly1.multiply(2, condense=True, inplace=True)  # condensing multiple Polygons gives a MultiPolygon
        testingdf = GeoDataFrame({'hex': testpoly1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testpoly1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with a list of different types
        testingdf = GeoDataFrame({'hex': testcollection1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testcollection1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

        # test with a geometry collection
        testcollection1.condense(inplace=True)  # condensing different types gives a GeometryCollection
        testingdf = GeoDataFrame({'hex': testcollection1.hexids}).set_index('hex').sort_index()
        inputdf = GeoDataFrame({'geometry': testcollection1.iter_shapes()})
        self.assertTrue(testingdf.index.equals(hexgrid.hexify_geodataframe(inputdf, 3).sort_index().index))

    def test_binning(self):
        """Tests the binning function.
        """

        # binning a single data set will return one for each hex bin
        inputdf = GeoDataFrame({'hex': testpoly1.hexids})
        testingdf = hexgrid.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals.pop(), 1)

        # ensuring duplicate hex ids by using duplicate geometries
        inputdf = GeoDataFrame({'hex': testpoly1.multiply(2).hexids})
        testingdf = hexgrid.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals.pop(), 2)

        inputdf = GeoDataFrame({'hex': testcollection1.hexids})
        testingdf = hexgrid.bin_by_hex(inputdf, lambda lst: len(lst), hex_field='hex', result_name='bin-op')
        vals = set(testingdf['bin-op'].unique())
        self.assertIn(2, vals)
        self.assertIn(4, vals)

    def test_validate_hexes(self):
        self.assertTrue(hexgrid.validate_hexes(testcollection1.hexids, get_sequence=False))
        self.assertFalse(hexgrid.validate_hexes(['123', '831208fffffffff'], get_sequence=False))

    def test_validate_dataframe_hexes(self):
        # test a group of known valid ids
        inputdf = GeoDataFrame({'hex': testcollection1.hexids})
        self.assertTrue(hexgrid.validate_dataframe_hexes(inputdf, hex_field='hex', store_validity=True))

        # test a group of mixed valid and invalid ids
        inputdf = GeoDataFrame({'hex': ['123', '831208fffffffff']})
        self.assertFalse(hexgrid.validate_dataframe_hexes(inputdf, hex_field='hex', store_validity=True))
        self.assertTrue('hex-validity' in inputdf.columns)  # ensure the validity was stored in a column
        self.assertTrue(True in inputdf['hex-validity'])  # ensure that one of the hex ids was valid
        self.assertListEqual(list(inputdf['hex-validity']), [False, True])


if __name__ == '__main__':
    unittest.main()
