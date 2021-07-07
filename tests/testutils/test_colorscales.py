import unittest
import geoviz.utils.colorscales as cli

roughly_equal = lambda x, y: round(x) == round(y)


def nested_colorscale_equal(cs1, cs2) -> bool:
    """Determines if two nested colorscales are roughly equal.

    :param cs1: The first nested colorscale
    :type cs1: Any
    :param cs2: The second nested colorscale
    :type cs2: Any
    :return: Whether the colorscales are roughly equal or not
    :rtype: bool
    """
    return all(roughly_equal(i1, i2) and c1 == c2 for (i1, c1), (i2, c2) in list(zip(cs1, cs2)))


class ColorscalesTestCase(unittest.TestCase):
    """Test cases for the colorscales module.
    """

    def test_get_scale_format(self):
        """Tests the module's ability to determine the format of a colorscale.

        Tests:
        Take colorscales in various formats and ensure the correct format
        was determined via the function.
        """

        testcs = 'Viridis'
        self.assertEqual(cli.get_cscale_format(testcs), 'string')
        testcs = {'Acceptable': 'rgb(255,255,255)'}
        self.assertEqual(cli.get_cscale_format(testcs), 'dict')
        testcs = [[0, 'rgb(255, 255, 1)'], [0.5, 'rgb(231, 231, 1)'], [1, 'rgb(90, 90, 91)']]
        self.assertEqual(cli.get_cscale_format(testcs), 'nested-iterable')
        testcs = ['rgb(255, 255, 1)', 'rgb(231, 231, 1)', 'rgb(90, 90, 91)']
        self.assertEqual(cli.get_cscale_format(testcs), 'iterable')
        testcs = set()
        self.assertEqual(cli.get_cscale_format(testcs), 'iterable')
        testcs = 125
        self.assertEqual(cli.get_cscale_format(testcs), 'unknown')

    def test_opacify(self):
        """Tests the module's ability to transform the opacity of a given colorscale.

        Tests:
        Take colorscales in various formats and ensure the correct opacity is within the resulting
        scale.
        """

        # test using colormap
        testcs = {
            'Acceptable': 'rgb(233, 241, 225)',
            'Exceeds': 'rgb(255, 255, 231)',
            'Failure': 'rgb(231, 231, 225)'
        }
        testalpha = 0.5
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for v in resultcs.values()))

        # test using nested colorscale (tuple and list)
        testcs = [[0, 'rgb(255, 255, 1)'], [0.5, 'rgb(231, 231, 1)'], [1, 'rgb(90, 90, 91)']]
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for _, v in resultcs))
        testcs = ((0, 'rgb(255, 255, 1)'), (0.5, 'rgb(231, 231, 1)'), (1, 'rgb(90, 90, 91)'))
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for _, v in resultcs))

        # test using regular colorscale (tuple and list)
        testcs = ['rgb(255, 255, 1)', 'rgb(231, 231, 1)', 'rgb(90, 90, 91)']
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for v in resultcs))
        testcs = 'rgb(255, 255, 1)', 'rgb(231, 231, 1)', 'rgb(90, 90, 91)'
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for v in resultcs))

        testcs = 'Viridis'
        resultcs = cli.configure_cscale_opacity(testcs, testalpha)
        self.assertTrue(all(str(testalpha) in v for v in resultcs))

    def test_discretize(self):
        """Tests the module's helper for making discrete colorscales.

        Input a bunch of colors and ensure the resulting colorscale is correct (spacing, colors).
        """
        testcenterp = 0.25
        testsizep = 0.25
        testcolors = ['red', 'blue', 'green', 'yellow', 'purple']

        self.assertTrue(nested_colorscale_equal(
            [[0.250001, 'red'], [0.5, 'red'], [0.500001, 'blue'], [0.75, 'blue'], [0.750001, 'green'], [1.0, 'green']],
            cli.discretize(testcolors, size_portion=testsizep, center_portion=testcenterp)))
        self.assertTrue(nested_colorscale_equal(
            [[0.0, 'red'], [0.25, 'red'], [0.250001, 'blue'], [0.5, 'blue'], [0.500001, 'green'], [0.75, 'green'],
             [0.750001, 'yellow'], [1.0, 'yellow']],
            cli.discretize(testcolors, size_portion=testsizep)))

    def test_discretize_sequential(self):
        """Tests the module's ability to discretize a single sequential scale.

        Tests:
        Input a bunch of colors and ensure the resulting colorscale is correct (spacing, colors).
        """
        with self.assertRaises(ValueError):
            cli.discretize_sequential([], 0, 1, discrete_size=1.0)

        self.assertTrue(nested_colorscale_equal([
            [0, 'red'], [0.25, 'red'],
            [0.250001, 'blue'], [0.5, 'blue'],
            [0.500001, 'green'], [0.75, 'green'],
            [0.750001, 'yellow'], [1.0, 'yellow']
        ], cli.discretize_sequential(['red', 'blue', 'green', 'yellow'], 0, 1, discrete_size=0.25)))

        self.assertTrue(nested_colorscale_equal([
            [0, 'rgb(255, 255, 255)'], [0.25, 'rgb(255, 255, 255)'],
            [0.250001, 'rgb(160, 160, 160)'], [0.5, 'rgb(160, 160, 160)'],
            [0.500001, 'rgb(180, 180, 180)'], [0.75, 'rgb(180, 180, 180)'],
            [0.750001, 'rgb(190, 190, 190)'], [1.0, 'rgb(190, 190, 190)']
        ], cli.discretize_sequential(['rgb(100, 100, 100)', 'rgb(255, 255, 255)', 'rgb(20, 20, 20)',
                                      'rgb(160, 160, 160)', 'rgb(180, 180, 180)', 'rgb(190, 190, 190)'],
                                     0, 1, choose_luminance=140, discrete_size=0.25)))

    def test_discretize_diverging(self):

        with self.assertRaises(ValueError):
            cli.discretize_diverging([], 0, 1, remove_middle=False)

        self.assertTrue(nested_colorscale_equal(
            [[0, 'blue'], [0.2, 'blue'],
             [0.2, 'red'], [0.4, 'red'],
             [0.4, 'green'], [0.6, 'green'],
             [0.6, 'yellow'], [0.8, 'yellow'],
             [0.8, 'purple'], [1.0, 'purple']
             ], cli.discretize_diverging(['blue', 'red', 'green', 'yellow', 'purple'],
                                         -2, 0, discrete_size=0.4, remove_middle=False)))

        self.assertTrue(nested_colorscale_equal(
            [
                [0, 'rgb(255, 255, 255)'], [0.2, 'rgb(255, 255, 255)'],
                [0.2, 'rgb(225, 225, 225)'], [0.4, 'rgb(225, 225, 225)'],
                [0.4, 'rgb(200, 200, 200)'], [0.6, 'rgb(200, 200, 200)'],
                [0.6, 'rgb(175, 175, 175)'], [0.8, 'rgb(175, 175, 175)'],
                [0.8, 'rgb(150, 150, 150)'], [1.0, 'rgb(150, 150, 150)']
            ], cli.discretize_diverging(
                ['rgb(255, 255, 255)', 'rgb(0, 0, 0)', 'rgb(225, 225, 225)', 'rgb(0, 0, 0)', 'rgb(200, 200, 200)',
                 'rgb(0, 0, 0)', 'rgb(175, 175, 175)', 'rgb(0, 0, 0)', 'rgb(150, 150, 150)'], -2, 0, discrete_size=0.4,
                remove_middle=False, center=-1.0, choose_luminance=100)
        ))

    def test_configure_color_opacity(self):
        """Tests the module's ability to conform the tranform rgb colors into rgba colors.

        Tests:
        Take multiple rgb colors and transform them. Ensure the result is correct (rgba with opacity).
        """
        self.assertEqual(cli.configure_color_opacity('rgb(245, 245, 245)', 0.4), 'rgba(245, 245, 245, 0.4)')
        self.assertEqual(cli.configure_color_opacity('red', 0.4), 'red')
        self.assertEqual(cli.configure_color_opacity('rgb(230, 230, 221)', 0.6), 'rgba(230, 230, 221, 0.6)')

    def test_try_get_scale(self):
        """Tests the module's ability to obtain a colorscale from any of Plotly's color modules.

        Normally Plotly Choropleths are restricted to sequential colorscales.

        Tests:
        Attempt to get color scales and verify that the operation was successful.
        """
        err = False
        try:
            cli.get_scale('gfhghg')
        except AttributeError:
            err = True
        self.assertTrue(err)
        self.assertTrue(cli.get_scale('Viridis'))

    def test_solid_scale(self):
        """Tests the module's ability to create a solid color scale from a color.

        Tests:
        Make a solid color scale and ensure the resulting color tuple is correct.
        """
        self.assertEqual(((0.0, 'red'), (1.0, 'red')), cli.solid_scale('red'))
        self.assertEqual(((0.0, 'rgb(255,255,255)'), (1.0, 'rgb(255,255,255)')), cli.solid_scale('rgb(255,255,255)'))

    def test_q(self):
        cli.discretize_cscale('Viridis', 'diverging', 0, 5, discrete_size=1, center=2.5, remove_middle=True)




if __name__ == '__main__':
    unittest.main()
