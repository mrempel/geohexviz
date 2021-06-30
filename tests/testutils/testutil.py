import unittest
from geopandas import GeoDataFrame
from geoviz.utils import util
from shapely.geometry import Point, GeometryCollection
from pandas import DataFrame
import pandas as pd


class UtilTestCase(unittest.TestCase):

    def test_combine_dataframes(self):
        leftdf = GeoDataFrame(dict(x=[1, 2, 3, 4, 5], y=[6, 7, 8, 9, 10],
                                   geometry=[Point(1, 1), Point(1, 2), Point(1, 3), Point(1, 4), Point(1, 5)]),
                              crs='EPSG:4326')
        leftdf.name = 'LEFT'

        rightdf = GeoDataFrame(dict(z=[4, 5, 6, 7, 8], y=[0, 0, 1, 2, 1],
                                    geometry=[Point(2, 1), Point(2, 2), Point(2, 3), Point(2, 4), Point(2, 5)]),
                               crs='EPSG:4326')
        rightdf.name = 'RIGHT'

        resultdf = util.combine_dataframes(leftdf, toadd=rightdf, as_index=True)
        expectedIndexL = ['LEFT'] * len(leftdf)
        expectedIndexL.extend(['RIGHT'] * len(rightdf))
        expectedIndexR = [x for x in range(len(leftdf))]
        expectedIndexR.extend([x for x in range(len(rightdf))])
        self.assertTrue(pd.MultiIndex.from_tuples(tuples=list(zip(expectedIndexL, expectedIndexR)),
                                                  names=('ORIGIN', 'ORIGIN_INDEX')).equals(resultdf.index))

        newdf = GeoDataFrame(
            dict(x=[0, 0, 0], w=[1, 3, 5], o=[9, 9, 9], geometry=[Point(3, 1), Point(3, 2), Point(3, 3)]),
            crs='EPSG:4326')
        newdf.name = 'NEW'

        resultdf = util.combine_dataframes(resultdf, toadd=newdf, as_index=True)
        expectedIndexL.extend(['NEW'] * len(newdf))
        expectedIndexR.extend([x for x in range(len(newdf))])
        self.assertTrue(pd.MultiIndex.from_tuples(tuples=list(zip(expectedIndexL, expectedIndexR)),
                                                  names=('ORIGIN', 'ORIGIN_INDEX')).equals(resultdf.index))

    def test_collapse_dataframe_by(self):
        testdf = DataFrame(dict(
            key=['A', 'A', 'A'],
            x=[1, [1, 2], [3, 4]],
            y=[2, 3, [4, 5]]
        ))

        resultdf = util.collapse_dataframe_by(testdf, by='key')
        self.assertListEqual(list(resultdf['OLEN'].values), [3, 2])

    def test_collapse_dataframe(self):
        testdf = DataFrame(dict(
            key=['A', 'A', 'A'],
            x=[1, [1, 2], [3, 4]],
            y=[2, 3, [4, 5]]
        ))
        resultdf = util.collapse_dataframe(testdf)
        self.assertListEqual(list(resultdf.iloc[0]['OLEN']), [3, 5, 4])


if __name__ == '__main__':
    unittest.main()
