# Forest Fires: GeoHexSimple JSON building walkthrough

To make this visualization the required layer and its properties are passed via the `hexbin_layer` object.

```json
{
	"hexbin_layer": {
		"data":"data.csv",
		"hexbin_info": {
			"hex_resolution":4,
			"binning_field": "FIRE_TYPE",
			"binning_fn": "best"
		},
		"manager": {
			"marker": {
				"line": {
					"width": 0.1
				}
			},
			"colorscale": "Dark24"
		}
	},
```

The `hexbin_layer` object has 3 members being `data`, `hexbin_info`, and `manager`.
Identical to the previous examples, the `data` member is now a full path to the data set containing the fire locations.
Next, the `hexbin_info` member, which is also an object, specifies how the data is to be hexagonally binned.
This is done through its 3 members: `binning_field`, `binning_fn`, and `hex_resolution`.
Similar to the second and third examples, in this example the data is to be binned by incident location and the value displayed is the most frequent category of fire in each hexagon.
The `binning_field` member of `hexbin_info` specifies that the display value be calculated from the `FIRE_TYPE` column, which is the column containing the category of fire.
The `binning_fn` member of `hexbin_info` then specifies that the `best` option be selected as the display value (the most frequent value).
The `hex_resolution` member of `hexbin_info` specifies the size of hexagon to be used.
This shows that the hexagon size can also be specified as a member of the `hexbin_info` object unlike the previous examples.
Finally, the `manager` specifies two properties that are passed into Plotly.
The first member, `marker` is used to specify the width of the lines used for the hexagons in the hexbin layer.
This is done through setting the `width` of the `marker`'s `line` property; the same properties were set in the World War 2 bombings example.
When set, the second member, `colorscale` specifies the colour scale to be used within the plot; in this case the colour scale is set to `Dark24`.
This property was also set in the Search and Recue example.


Since the region of focus in this example is USA, a region layer containing the USA landmass is added to the plot via the `regions` object.

```json
	"regions": {
		"sample_Region_USA": {
			"data": "UNITED STATES OF AMERICA"
		}
	},
```

`sample_Region_USA` is the object defining this region layer.
The data member of the region layer is set to `UNITED STATES OF AMERICA`.


Now extended grid layers are added to fill the gaps within the data and form a continuous grid.
Once again, this is done by adding the `grids` object to the JSON file.

```json
	"grids": {
		"sample_Grid_USA": {
			"data": "UNITED STATES OF AMERICA",
			"convex_simplify": true
		}
	},
```

Since the data spans the United States of America, we declare a grid layer that also spans this region.
This grid layer is defined through the object `sample_Grid_USA`, whose `data` member is set to `UNITED STATES OF AMERICA`.


Next the Plotly properties of the grid layers are set to match the Plotly properties of the hexbin layer.
This is done by adding the `grid_manager` object to the JSON file.

```json
	"grid_manager": {
		"marker": {
			"line": {
				"width": 0.1
			}
		}
	},
```

Next, some properties of the legend are set for aesthetic purposes.
The properties of the legend are stored within the internal figure's layout properties.
In order to interact with the internal figure's layout, the `figure_layout` object is added to the JSON file.

```json
	"figure_layout": {
		"legend": {
			"x": 0.8043,
			"bordercolor": "black",
			"borderwidth": 1,
			"font": {
				"size": 8
			}
		}
	},
```

The properties of the legend are set by adding the `legend` member/object to the `figure_layout` object.
The `legend` property controls the different features of the legend, such as width, legend item sizing, and legend title; for the full list of input options, see \citet{plotly-figure-layout-legend-docs}.
The `legend` object has four members which control positioning `x`, the colour of the border `bordercolor`, the width of the border `borderwidth`, and the size of the font (controlled through the `size` member of the `font` property).


Next, a set of functions are specified in the `functions` object of the JSON file.

```json
	"functions": {
		"clip_layers": {
			"clip": "hexbin+grids",
			"to": "regions"
		},
		"adjust_focus": {
			"on": "hexbin",
			"buffer_lat": [0,15],
			"rot_buffer_lon": -8
		}
	},
```

The first function is the `clip_layers`, which specifies that hexbin and grid layers be clipped to region layers; this same function is used in the Search and Rescue, and World War 2 examples.
The second function is the `adjust_focus`, which specifies that the plot be focused on the hexbin layer (but slightly shifted); used in the Search and Rescue example.


Finally, the output location of the visualization is specified through the `output`.
This step is identical to the examples shown in all other examples.

```json
	"output": {
		"filepath": "output_visualization.pdf",
		"crop_output": true
	}
}
```

The full JSON file is given below.
The Python module translation of this JSON is given in `python_walkthrough.ipynb`, and `python_walkthrough.py`.

```json
{
	"hexbin_layer": {
		"data":"data.csv",
		"hexbin_info": {
			"hex_resolution":4,
			"binning_field": "FIRE_TYPE",
			"binning_fn": "best"
		},
		"manager": {
			"marker": {
				"line": {
					"width": 0.1
				}
			},
			"colorscale": "Dark24"
		}
	},
	"regions": {
		"sample_Region_USA": {
			"data": "UNITED STATES OF AMERICA"
		}
	},
	"grids": {
		"sample_Grid_USA": {
			"data": "UNITED STATES OF AMERICA",
			"convex_simplify": true
		}
	},
	"figure_layout": {
		"legend": {
			"x": 0.8043,
			"bordercolor": "black",
			"borderwidth": 1,
			"font": {
				"size": 8
			}
		}
	},
	"functions": {
		"clip_layers": {
			"clip": "hexbin+grids",
			"to": "regions"
		},
		"adjust_focus": {
			"on": "hexbin",
			"buffer_lat": [0,15],
			"rot_buffer_lon": -8
		}
	},
	"output": {
		"filepath": "output_visualization.pdf",
		"crop_output": true
	}
}
```
