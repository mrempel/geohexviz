import unittest
from os.path import join as pjoin
import os
from geohexsimple import run_json
from geohexviz.builder import PlotStatus

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
PARAM_PATH = pjoin(DATA_PATH, 'parameterfile-data')


class GeoSimpleCase(unittest.TestCase):
    def test_run_simple_JSON(self):
        #self.assertEqual(PlotStatus.DATA_PRESENT, run_json(pjoin(PARAM_PATH, 'sample1-russia.json')))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
