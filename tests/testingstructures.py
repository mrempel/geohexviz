from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString

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