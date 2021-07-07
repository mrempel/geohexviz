import unittest
import random
import numpy as np
import geoviz.utils.plot_util as butil


class PlotUtilTestCase(unittest.TestCase):
    """Tests the functions within the plot_util module.
    """

    def test_format_latex_exp10(self):
        """Tests the module's ability to format a base 10 latex exponent.

        Tests:
        Invoke the function for each exponent type. Ensure the result is correct.
        """
        with self.assertRaises(ValueError):
            butil.format_latex_exp10(1, 'l')
        self.assertEqual('$10^1$', butil.format_latex_exp10(1, '^'))
        self.assertEqual('$1E10$', butil.format_latex_exp10(10, 'E'))
        self.assertEqual('$1e10$', butil.format_latex_exp10(10, 'e'))
        self.assertEqual('$10**2$', butil.format_latex_exp10(2, '*'))
        self.assertEqual('$1000$', butil.format_latex_exp10(3, 'r'))

    def test_format_html_exp10(self):
        """Tests the module's ability to format a base 10 html exponent.

        Tests:
        Invoke the function for each exponent type. Ensure the result is correct.
        """
        with self.assertRaises(ValueError):
            butil.format_html_exp10(1, 'l')
        self.assertEqual('<span>10<sup>1</sup></span>', butil.format_html_exp10(1, '^'))
        self.assertEqual('<span>1E10</span>', butil.format_html_exp10(10, 'E'))
        self.assertEqual('<span>1e10</span>', butil.format_html_exp10(10, 'e'))
        self.assertEqual('<span>10**2</span>', butil.format_html_exp10(2, '*'))
        self.assertEqual('<span>10^3</span>', butil.format_html_exp10(3, 'r'))
        self.assertEqual('<span>1000</span>', butil.format_html_exp10(3, 'n'))

    def test_format_raw_exp10(self):
        """Tests the module's ability to format a base 10 raw exponent.

        Tests:
        Invoke the function for each exponent type. Ensure the result is correct.
        """
        with self.assertRaises(ValueError):
            butil.format_raw_exp10(1, 'l')
        self.assertEqual('10^1', butil.format_raw_exp10(1, '^'))
        self.assertEqual('1E10', butil.format_raw_exp10(10, 'E'))
        self.assertEqual('1e10', butil.format_raw_exp10(10, 'e'))
        self.assertEqual('10**2', butil.format_raw_exp10(2, '*'))
        self.assertEqual('1000', butil.format_raw_exp10(3, 'r'))

    def test_logify_info(self):
        """Tests the module's ability to retrieve a dict of logarithmic scale info for a set of values.

        Tests:
        Generate values and invoke the function. Ensure that the logarithmic info is correct (min, max, logged values).
        """
        with self.assertRaises(ValueError):
            butil.logify_info([])

        values = [random.randint(1, 100000) for _ in range(1000)]
        result = butil.logify_info(values, minmax_rounding=3, include_min=True, include_max=True, max_prefix='max',
                                   min_prefix='min')
        self.assertTrue(all(y == np.log10(x) for x, y in zip(result['original-values'], result['logged-values'])))
        lmin, lmax = min(result['logged-values']), max(result['logged-values'])
        self.assertEqual(round(lmin, 3), result['scale-min'])
        self.assertEqual(round(lmax, 3), result['scale-max'])
        self.assertTrue(all(x in result['scale-dict'] for x in range(int(lmin), int(lmax) + 1)))
        self.assertTrue(any('max' in x for x in result['scale-dict'].values()))  # check for max prefix
        self.assertTrue(any('min' in x for x in result['scale-dict'].values()))  # check for min prefix

        result = butil.logify_info(values, minmax_rounding=3, include_min=True, include_max=True, max_prefix='max',
                                   min_prefix='min', include_predecessors=True)
        self.assertTrue(all(y == np.log10(x) for x, y in zip(result['original-values'], result['logged-values'])))
        lmin, lmax = min(result['logged-values']), max(result['logged-values'])
        self.assertEqual(round(lmin, 3), result['scale-min'])
        self.assertEqual(round(lmax, 3), result['scale-max'])
        self.assertTrue(all(x in result['scale-dict'] for x in range(1, int(lmax) + 1)))  # for include predecessors
        self.assertTrue(any('max' in x for x in result['scale-dict'].values()))  # check for max prefix
        self.assertTrue(any('min' in x for x in result['scale-dict'].values()))  # check for min prefix


if __name__ == '__main__':
    unittest.main()
