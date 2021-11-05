"""
Forest Fires

This tutorial provides two methods of creating the visualization for example two,
both of which pertain to usage within Python code.
"""

# this file needs to be run for each .csv file (each year)
# Method 2a: Using GeoHexSimple's functions to run a properties file
from geohexviz.utils.file import run_file

run_file("<path to example3.json>")  # JSON file works

# Method 2b: Using GeoHexViz's functions to make a plot from scratch
from geohexviz.builder import PlotBuilder

myBuilder = PlotBuilder()

# set hexbin layer
myBuilder.set_hexbin(
    data="<path to data.csv>",
    hexbin_info=dict(
        hex_resolution=4,
        binning_field="FIRE_TYPE",
        binning_fn="best"
    ),
    manager=dict(
        marker=dict(
            line=dict(
                width=0.1
            )
        ),
        colorscale="Dark24"
    )
)

# add region layers
myBuilder.add_region(
    name="sample_Region_USA",
    data="UNITED STATES OF AMERICA"
)

# add grid layers
myBuilder.add_grid(
    name="sample_Grid_USA",
    data="UNITED STATES OF AMERICA"
)

# alter figure layout
myBuilder.update_figure(
    layout=dict(
        legend=dict(
            x=0.8043,
            bordercolor="black",
            borderwidth=1,
            font=dict(
                size=8
            )
        )
    )
)

# invoke functions
myBuilder.clip_layers(
    clip="hexbin+grids",
    to="regions"
)
myBuilder.adjust_focus(
    on="hexbin+grids",
    buffer_lat=[0, 15],
    rot_buffer_lon=-8
)

# finalize and output
myBuilder.finalize()
myBuilder.output(
    filepath="<path to output (.pdf)>",
    crop_output=True
)