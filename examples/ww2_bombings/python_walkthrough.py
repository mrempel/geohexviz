"""
World War 2 Bombings

This tutorial provides two methods of creating the visualization for example two,
both of which pertain to usage within Python code.
Note that this needs to be run three different times for each .csv file (for each year).
"""

# this file needs to be run for each .csv file (each year)
# Method 2a: Using GeoHexSimple's functions to run a properties file
from geohexsimple import run_file

run_file("<path to example3.json>")  # JSON file works
run_file("<path to example3.yml>")  # YAML file works

# Method 2b: Using GeoHexViz's functions to make a plot from scratch
from geohexviz.builder import PlotBuilder

myBuilder = PlotBuilder()

# set hexbin layer
myBuilder.set_hexbin(
    data="<path to data-1943.csv, data-1944.csv, data-1945.csv>",
    hexbin_info=dict(
        binning_field="high_explosives_weight_tons",
        binning_fn="sum"
    ),
    hex_resolution=4,
    manager=dict(
        marker=dict(
            line=dict(width=0.45)
        ),
        colorscale="Viridis",
        colorbar=dict(
            x=0.82
        )
    )
)

# add region layers
myBuilder.add_region(
    name="sample_Region_EUROPE",
    data="EUROPE"
)

# add grid layers
myBuilder.add_grid(
    name="sample_Grid_EUROPE",
    data="EUROPE",
    convex_simplify=True
)
myBuilder.add_grid(
    name="sample_Grid_RUSSIA",
    data="RUSSIA",
    convex_simplify=True
)

# update grid manager
myBuilder.update_grid_manager(
    marker=dict(
        line=dict(width=0.45)
    )
)

# update figure geos
myBuilder.update_figure(
    geos=dict(
        lataxis=dict(
            range=[35, 58]
        ),
        lonaxis=dict(
            range=[0, 43]
        ),
        projection=dict(
            rotation=dict(
                lat=46.63321662159487,
                lon=11.21560455920799
            )
        )
    )
)

# invoke functions
myBuilder.clip_layers(
    clip="hexbin+grids",
    to="regions"
)
myBuilder.logify_scale(
    exp_type="r"
)
# * Unlike JSON input mechanism, in a module adjust\_focus is not
# * invoked by default, the user has to invoke it

# finalize and output
myBuilder.finalize()
myBuilder.output(
    filepath="<path to output (.pdf)>",
    crop_output=True
)