# GeoViz

GeoVis is a package for the simple and repeatable visualization of hexagon-ally binned data sets.\
The package's main feature is a PlotBuilder class which utilizes tools to hexagon-ally bin your dataset and then display it.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install GeoHexViz.

```bash
pip install geoviz
```

## Usage
```python
from pandas import DataFrame
from geoviz.builder import PlotBuilder

# Creating an example dataset
inputdf = DataFrame(dict(
    latitude=[17.57, 17.57, 17.57, 19.98, 19.98, 46.75],
    longitude=[10.11, 10.11, 10.12, 50.55, 50.55, 31.17],
    value=[120, 120, 120, 400, 400, 700]
))

# Instantiating builder
builder = PlotBuilder()
builder.set_main(inputdf, hexbin_info=dict(binning_fn='sum', binning_field='value'))

builder.build_plot(raise_errors=False)
builder.display_figure(clear_figure=True)

# A mapbox map
builder.set_mapbox('<ACCESS TOKEN>')
builder.build_plot()
builder.display_figure(clear_figure=True)
```

###Behind the Scenes
When the main dataset is passed into the builder, the data is processed in the following steps:

Data:

| index | lats  | lons  | value |
|-------|-------|-------|-------|
| 0     | 17.57 | 10.11 | 120   |
| 1     | 17.57 | 10.11 | 120   |
| 2     | 17.57 | 10.12 | 120   |
| 3     | 19.98 | 50.55 | 400   |
| 4     | 19.98 | 50.55 | 400   |
| 5     | 46.75 | 31.17 | 700   |

1) Coordinate columns are converted into geometry (if applicable)

| index | value | geometry            |
|-------|-------|---------------------|
| 0     | 120   | POINT(17.57, 10.11) |
| 1     | 120   | POINT(17.57, 10.11) |
| 2     | 120   | POINT(17.57, 10.12) |
| 3     | 400   | POINT(19.98, 50.55) |
| 4     | 400   | POINT(19.98, 50.55) |
| 5     | 700   | POINT(46.75, 31.17) |

2) Hex cells are then placed over the data

| hex             | value | geometry            |
|-----------------|-------|---------------------|
| 83595afffffffff | 120   | POINT(17.57, 10.11) |
| 83595afffffffff | 120   | POINT(17.57, 10.11) |
| 83595afffffffff | 120   | POINT(17.57, 10.12) |
| 835262fffffffff | 400   | POINT(19.98, 50.55) |
| 835262fffffffff | 400   | POINT(19.98, 50.55) |
| 831e5dfffffffff | 700   | POINT(46.75, 31.17) |
(resolution = 3)

3) The data is grouped together by hex, and hex geometry is added

| hex             | items     | value_field | geometry                                          |
|-----------------|-----------|-------------|---------------------------------------------------|
| 831e5dfffffffff | (5)       |      1      | POLYGON ((30.57051 46.80615, 30.47843 46.19931... |
| 835262fffffffff | (3, 4)    |      2      | POLYGON ((49.90903 20.19437, 49.74835 19.60088... |
| 83595afffffffff | (0, 1, 2) |      3      | POLYGON ((9.44614 17.39197, 9.49704 16.75205, ... |
(binning function = num. of occurrences within a hex)

When the data is eventually plotted, a GeoJSON format of the data is passed
alongside plotly properties are passed to the Plotly graphing library.

## Contributing
For major changes, please open an issue first to discuss what you would like to change.

## License