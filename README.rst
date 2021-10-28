GeoHexViz
=========

GeoHexViz is a package for the simple and repeatable visualization of
hexagon-ally binned data sets.
The package's main feature is a PlotBuilder class which utilizes tools
to hexagon-ally bin your dataset and then display it.

Command Line / JSON Usage
-------------------------

The GeoHexViz distribution includes a module that can allow the reading
of JSON files for quick and easy plots.

.. code:: json

    {
      "main_dataset": {
        "data": "<sample csv file>",
        "hex_resolution": 4
      },
      "output_figure": {
        "filepath": "<sample filepath>",
        "width": 600,
        "height": 400
      },
      "display_figure": true
    }

running the JSON script will allow you to input a JSON file via command
line.

You may also use the functions that allow the executing of a JSON file
from a python module.

.. code:: python

    from geohexsimple import run_json

    run_json("<filepath here>")

Python Module Usage (PlotBuilder)
---------------------------------

.. code:: python

    from pandas import DataFrame
    from geohexviz.builder import PlotBuilder

    # Creating an example dataset
    inputdf = DataFrame(dict(
        latitude=[17.57, 17.57, 17.57, 19.98, 19.98, 46.75],
        longitude=[10.11, 10.11, 10.12, 50.55, 50.55, 31.17],
        value=[120, 120, 120, 400, 400, 700]
    ))

    # Instantiating builder
    builder = PlotBuilder()
    builder.set_hexbin(inputdf, hexbin_info=dict(binning_fn='sum', binning_field='value'))

    builder.finalize(raise_errors=False)
    builder.display(clear_figure=True)

    # A mapbox map
    builder.set_mapbox('<ACCESS TOKEN>')
    builder.finalize()
    builder.display(clear_figure=True)

Behind the Scenes
~~~~~~~~~~~~~~~~~

When the main dataset is passed into the builder, the data is processed
in the following steps:

Data:

+-------+-------+-------+-------+
| index |  lats |  lons | value |
+=======+=======+=======+=======+
|   0   | 17.57 | 10.11 |  120  |
+-------+-------+-------+-------+
|   1   | 17.57 | 10.11 |  120  |
+-------+-------+-------+-------+
|   2   | 17.57 | 10.12 |  120  |
+-------+-------+-------+-------+
|   3   | 19.98 | 50.55 |  400  |
+-------+-------+-------+-------+
|   4   | 19.98 | 50.55 |  400  |
+-------+-------+-------+-------+
|   5   | 46.75 | 31.17 |  700  |
+-------+-------+-------+-------+

1) Coordinate columns are converted into geometry (if applicable)

+-------+-------+---------------------+
| index | value |       geometry      |
+=======+=======+=====================+
|   0   |  120  | POINT(17.57, 10.11) |
+-------+-------+---------------------+
|   1   |  120  | POINT(17.57, 10.11) |
+-------+-------+---------------------+
|   2   |  120  | POINT(17.57, 10.12) |
+-------+-------+---------------------+
|   3   |  400  | POINT(19.98, 50.55) |
+-------+-------+---------------------+
|   4   |  400  | POINT(19.98, 50.55) |
+-------+-------+---------------------+
|   5   |  700  | POINT(46.75, 31.17) |
+-------+-------+---------------------+

2) Hex cells are then placed over the data

+-----------------+-------+---------------------+
|       hex       | value |       geometry      |
+=================+=======+=====================+
| 83595afffffffff |  120  | POINT(17.57, 10.11) |
+-----------------+-------+---------------------+
| 83595afffffffff |  120  | POINT(17.57, 10.11) |
+-----------------+-------+---------------------+
| 83595afffffffff |  120  | POINT(17.57, 10.11) |
+-----------------+-------+---------------------+
| 835262fffffffff |  400  | POINT(19.98, 50.55) |
+-----------------+-------+---------------------+
| 835262fffffffff |  400  | POINT(19.98, 50.55) |
+-----------------+-------+---------------------+
| 831e5dfffffffff |  700  | POINT(46.75, 31.17) |
+-----------------+-------+---------------------+
(hex resolution = 3)

3) The data is grouped together by hex, and hex geometry is added

+-----------------+---------------+-------------+---------------------------------------------------+
|       hex       |     items     | value_field |                      geometry                     |
+=================+===============+=============+===================================================+
| 83595afffffffff | (120,120,120) |     360     | POLYGON ((30.57051 46.80615, 30.47843 46.19931... |
+-----------------+---------------+-------------+---------------------------------------------------+
| 835262fffffffff |   (400, 400)  |     800     | POLYGON ((49.90903 20.19437, 49.74835 19.60088... |
+-----------------+---------------+-------------+---------------------------------------------------+
| 831e5dfffffffff |     (700)     |     700     | POLYGON ((9.44614 17.39197, 9.49704 16.75205, ... |
+-----------------+---------------+-------------+---------------------------------------------------+
(binning function = sum of grouped values)

When the data is eventually plotted, a GeoJSON format of the data is
passed alongside plotly properties are passed to the Plotly graphing
library.

Installation
------------

As a prerequisite, the user must install the ``GeoPandas`` library before
installing ``GeoHexViz``.
This can be done easily in an Anaconda environment by doing running the
following command:

.. code:: bash

    conda install -c conda-forge geopandas

More information can be seen in the GeoPandas official documentation.


As of right now the GeoHexViz package can be cloned on GitHub, and
install by using the ``setup.py`` file.
This can be done by navigating to the folder containing the ``setup.py`` file,
and running the following command:

.. code:: bash

    python setup.py install

Further Documentation
---------------------

There is further documentation contained within the Reference Document
published alongside this software package, which is available {HERE}.
The official API documentation is also available {HERE}.

Acknowledgements
----------------

Thank you to Nicholi Shiell for his input in testing, and providing
advice for the development of this package.

Limitations
-----------

This package uses GeoJSON format to plot data sets. With GeoJSON comes
difficulties when geometries cross the 180th meridian . The issue
appears to cause a color that bleeds through the entire plot and leaves
a hexagon empty. In the final plot, this issue may or may not appear as
it only occurs at certain angles of rotation. In this package a simple
solution to the problem is implemented, in the future it would be best
to provide a more robust solution. The solution that is used works
generally, however, when hexagons containing either the north or south
pole are present, the solution to the 180th meridian issue persists.
This pole issue can be seen below.

There also exists some issues with the generation of discrete color
scales under rare circumstances. These circumstances include generating
discrete color scales with not enough hues to fill the scale, and
generating diverging discrete colorscales with the center hue in a weird
position. These issues have been noted and will be fixed in the near
future.

There exists issues with the positioning and height of the color bar
with respect to the plot area of the figure. Although the user is
capable of altering the dimensions and positioning of the color bar,
this should be done automatically as it is a common feature of
publication quality choropleth maps.

Contributing
------------

For major changes, please open an issue first to discuss what you would
like to change.

Contact
-------

For any questions, feedback, bug reports, feature requests, etc please
first present your thoughts via GitHub issues. For further assistance
please contact tony.azp25@gmail.com.

Copyright and License
---------------------

Copyright (c) Her Majesty the Queen in Right of Canada, as represented
by the Minister of National Defence, 2021.
