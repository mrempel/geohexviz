# Method 1: Using GeoHexSimple's functions to run a properties file
from geohexsimple import run_file

run_file("<path to example1.json>")  # JSON file works
run_file("<path to example1.yml>")  # YAML file works

# Method 2: Using GeoHexViz's functions to make a plot from scratch
from geohexviz.builder import PlotBuilder

myBuilder = PlotBuilder()

# set hexbin layer
myBuilder.set_hexbin(
    data="<path to data.csv>",
    hex_resolution=4,
    manager=dict(
        colorscale="Viridis",
        colorbar=dict(
            x=0.8325
        )
    )
)

# add region layers
myBuilder.add_region(
    name="sample_Region_CANADA",
    data="CANADA"
)

# add grid layers
myBuilder.add_grid(
    name="sample_Grid_CANADA",
    data="CANADA"
)

# invoke functions
myBuilder.clip_layers(
    clip="hexbin+grids",
    to="regions"
)
myBuilder.adjust_focus(
    on="regions",
    buffer_lat=[0, 3]
)
myBuilder.logify_scale(
    exp_type="r"
)

# finalize and output
myBuilder.finalize()
myBuilder.output(
    filepath="<path to output (.pdf)>",
    crop_output=True
)
