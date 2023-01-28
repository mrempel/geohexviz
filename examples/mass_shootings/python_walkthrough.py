"""
Mass Shootings

This tutorial provides two methods of creating the visualization for example two,
both of which pertain to usage within Python code.
"""

# Method 2a: Using GeoHexSimple's functions to run a properties file
from geohexviz.utils.file import run_file

run_file("json_structure.json")  # JSON file works

# Method 2b: Using GeoHexViz's functions to make a plot from scratch
from geohexviz.builder import PlotBuilder

myBuilder = PlotBuilder()

# set hexbin layer
myBuilder.set_hexbin(
    data="data.csv",
    hex_resolution=3,
    hexbin_info=dict(
        binning_field="killed_injured",
        binning_fn="sum"
    ),
    manager=dict(
        colorbar=dict(
            x=0.8365
        )
    )
)

# add region layers
myBuilder.add_region(
    name="sample_Region_USA",
    data="UNITED STATES OF AMERICA"
)

myBuilder.add_point(
    name="sample_Point_EPICENTERS",
    data="data-epicenters.csv",
    manager=dict(
        textposition=[
            "top center",
            "top center",
            "middle right",
            "top center",
            "top left",
            "bottom right",
            "top center",
            "top center",
            "top center",
            "top center"
        ],
        marker=dict(
            symbol="square-dot",
            size=4,
            line=dict(
                width=0.5
            )
        )
    )
)

# invoke functions
myBuilder.remove_empties()
myBuilder.adjust_focus(
    on="hexbin+grids",
    buffer_lat=[0, 15],
    rot_buffer_lon=-8
)
myBuilder.logify_scale(
    exp_type="r"
)

# finalize and output
myBuilder.finalize()
myBuilder.output(
    filepath="output_visualization.pdf",
    crop_output=True
)
