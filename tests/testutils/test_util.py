import unittest
from pandas import DataFrame
import geohexviz.utils.util as util


class UtilTestCase(unittest.TestCase):
    """Test cases for the util module.
    """

    def test_dict_deep_update(self):
        """Tests the module's ability to deep update a dictionary.

        Tests:
        Take a dict and update it with one of similar structure. Ensure that
        the resulting dict has not been overwritten shallowly.
        """
        dct = {
            'colorbar': {
                'tickvals': [1, 2, 3, 4],
                'ticktext': ['10', '100', '1000', '10000']
            }
        }
        up = {
            'colorbar': {
                'ticktext': ['1', '2', '3', '4']
            }
        }
        util.dict_deep_update(dct, up)
        self.assertEqual(dct, {
            'colorbar': {
                'tickvals': [1, 2, 3, 4],
                'ticktext': ['1', '2', '3', '4']
            }
        })

    def test_get_sorted_occurrences(self):
        """Tests the module's ability to best occurrence in a list.

        Tests:
        Take a list and get the best occurrence from it. Ensure that the result is correct.
        """
        result = util.get_sorted_best(['allow', 'allow', 'deny', 'deny', 'permit'], selector=['deny', 'allow'])
        self.assertEqual('deny', result)
        result = util.get_sorted_best(['allow', 'allow', 'deny', 'deny', 'permit'], selector=['allow', 'deny'])
        self.assertEqual('allow', result)
        result = util.get_sorted_best(['allow', 'allow', 'deny', 'deny', 'permit'], selector=['deny', 'allow'],
                                      allow_ties=True, join_ties=True)
        self.assertEqual(', '.join(['allow', 'deny']), result)
        result = util.get_sorted_best(['allow', 'allow', 'deny', 'deny', 'permit'], selector=['deny', 'allow'],
                                      allow_ties=True, join_ties=False)
        self.assertEqual('tie', result)
        result = util.get_sorted_best(['allow', 'allow', 'deny', 'deny', 'permit'], selector=['deny', 'allow'],
                                      reverse=False)
        self.assertEqual('permit', result)

    def test_get_column_or_default(self):
        """Tests the module's ability to get a column from a dataframe or default.

        Tests:
        Take a dataframe and check get columns using the function. Use both columns
        inside of the dataframe and columns that don't exist. Ensure the column exists or doesn't.
        """
        testdf = DataFrame(dict(x=[1, 2, 3, 4, 5], y=[9, 9, 9, 9, 9]))
        self.assertTrue(util.get_column_or_default(testdf, 'z', default_val=True))
        self.assertTrue(util.get_column_or_default(testdf, 'y', default_val=None) is not None)

    def test_get_column_type(self):
        """Test's the modules ability to get the type of a column from a dataframe.

        Tests:
        Take a dataframe and checks the type of the column using the function. Ensure the type
        is correct.
        """
        testdf = DataFrame(dict(x=[1, 2, 3, 4, 5], y=['9', '9', '9', '9', '9'], z=[[], [], [], [], []]))
        print(testdf)
        self.assertEqual(util.get_column_type(testdf, 'x'), 'NUM')
        self.assertEqual(util.get_column_type(testdf, 'y'), 'STR')
        self.assertEqual(util.get_column_type(testdf, 'z'), 'UNK')

    def test_generate_dataframe_random_ids(self):
        testdf = DataFrame(dict(x=[i for i in range(0, 100)]))
        util.generate_dataframe_random_ids(testdf)
        # continue...?


if __name__ == '__main__':
    unittest.main()
