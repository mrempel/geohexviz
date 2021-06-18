# GeoViz

GeoVis is a package for the simple and repeatable visualization of hexagon-ally binned data sets.\
The package's main feature is a PlotBuilder class which utilizes tools to hexagon-ally bin your dataset and then display it.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install geoviz
```

## Usage
```python
from pandas import DataFrame
from geoviz.builder import PlotBuilder

# Creating an example dataset
inputdf = DataFrame(dict(
    lats=[17.57, 17.57, 17.57, 19.98, 19.98, 46.75],
    lons=[10.11, 10.11, 10.12, 50.55, 50.55, 31.17],
    value=[120, 120, 120, 400, 400, 700]
))

# Instantiating builder
builder = PlotBuilder()
builder.set_main(inputdf, latitude_field='lats', longitude_field='lons',
                 binning_field='value', binning_fn=sum)

builder.build_plot()

# A mapbox map
builder.set_mapbox('<ACCESS TOKEN>')
builder.build_plot()
```

## Contributing
For major changes, please open an issue first to discuss what you would like to change.

## License