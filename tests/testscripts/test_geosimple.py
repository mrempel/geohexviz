import unittest
from os.path import join as pjoin
import os
from scripts.geosimple import util
from geoviz.builder import PlotStatus

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')
PARAM_PATH = pjoin(DATA_PATH, 'parameterfile-data')


class GeoSimpleCase(unittest.TestCase):
    def test_run_simple_JSON(self):
        self.assertEqual(PlotStatus.DATA_PRESENT, util.run_json(pjoin(PARAM_PATH, 'sample1-russia.json')))


if __name__ == '__main__':
    unittest.main()
