# World War 2 bombings: GeoHexSimple JSON building walkthrough

To make this visualization the required layer and its properties are passed via `hexbin_layer` object.

```json
{
	"hexbin_layer": {
		"data":"<path to data-1943.csv, data-1944.csv, data-1945.csv> (run for each year)",
		"latitude_field": "target_latitude",
		"longitude_field": "target_longitude",
		"hexbin_info": {
			"binning_field": "high_explosives_weight_tons",
			"binning_fn": "sum"
		},
		"hex_resolution":4,
		"manager": {
			"marker": {
				"line": {"width": 0.45}
			},
			"colorscale": "Viridis",
			"colorbar": {
				"x": 0.82
			}
		}
	},
```

Similar to the Mass Shootings example, the `hexbin_layer` object has four members.
The `data` member is now a full path to the data set containing the bombing locations.
Next, the `hexbin_info` member is an object specifying how the data is to be binned through its members.
The `binning_field` member of `hexbin_info` specifies that the data be grouped by `high_explosives_weight_tons`- a column containing the weight of bombs dropped at each incident location.
Once again, the `binning_fn` member of `hexbin_info` specifies that the grouped data be summed to retrieve the display value.
Next, the `manager` specifies three members that are passed into Plotly being `marker`, `colorscale`, and `colorbar`.
The `marker` member controls drawing properties for its associated layer (hexbin layer in this case) such as opacity, and other line properties.
In this example the `marker` member is an object specifying that the line width be set.
It does this through the `line` member which is also an object controlling the properties of line colour and width.
In this example, the property `width` is being set which controls line width.


Since the region of focus in this example is Europe, a region layer containing the European landmass is added to the plot via the `regions` object.

```json
	"regions": {
		"sample_Region_EUROPE": {
			"data": "EUROPE"
		}
	},
```

The `sample_Region_EUROPE` is the object defining this region layer.
The data member of the region layer is set to `EUROPE`.


Now extended grid layers are added to fill the gaps within the data and form a continuous grid.
Once again, this is done by adding the `grids` object to the JSON file.

```json
	"grids": {
		"sample_Grid_EUROPE": {
			"data": "EUROPE",
			"convex_simplify": true
		},
		"sample_Grid_RUSSIA": {
			"data": "RUSSIA"
		}
	},
```

Since the data spans the European region, we declare a grid layer that also spans this region.
This grid layer is defined through the object `sample_Grid_EUROPE`, whose `data` member is set to `EUROPE`.
It becomes evident that if the grid layer `sample_Grid_RUSSIA` is not present, then there are few hexagons present near Russia.


Next, since the line thickness for the hexbin layer has been altered, the line thickness for all grid layers must be the same.
This change is made by adding the `grid_manager`.

```json
	"grid_manager": {
		"marker": {
			"line": {"width": 0.45}
		}
	},
```

The properties set for this manager's line thickness are identical to those set in the \pref{manager}{examplequant-ww2-hexbin-manager} of the hexbin layer.


Next, since using the `adjust_focus` function does not provide the necessary focus for this plot easily, it is set manually.
To do this, the geo layout properties (Plotly) needs to set; this is done via adding the `figure_geos`.

```json
	"figure_geos": {
		"lataxis_range": [35, 58],
		"lonaxis_range": [0, 43],
		"projection_rotation": {
			"lat": 46.63321662159487,
			"lon": 11.21560455920799
		}
	},
```

The default projection type of GeoHexViz is the orthographic projection supplied by Plotly.
In order to obtain the correct focus for this type of projection there are three properties that need to be set.
These properties are the latitude axis range, longitude axis range, and projection rotation.
The latitude axis range and longitude axis range specify the range of latitudes and longitudes that appear in the figure once generated.
The projection rotation makes the globe rotate to the specified coordinates.
First, to set the latitude axis range, the `lataxis` member is added to the `figure_geos` object.
The `lataxis` object controls many properties for the latitude axis displayed on the figure, such as grid width and grid colour.
For this example, the `range` property of the `lataxis` is set to the range to be displayed in the figure, which is `[35, 58]` or from 35 degrees to 58 degrees.
Similarly, to set the longitude axis range, the `lonaxis` member is added to the `figure_geos` object.
The `lonaxis` object controls many properties for the longitude axis displayed on the figure.
For this example, the `range` property of the `lonaxis` is set to the range to be displayed in the plot, which is `[0, 43]` or from 0 degrees to 43 degrees.
Finally, the projection rotation is set via adding the `projection` member to the `figure_geos` object.
The `projection` object controls many properties for the projection that the data be displayed on, such as the type of projection used, the tilt of the projection, and the scale of the projection.
For this example the `rotation` property of the `projection` has its `lat`, and `lon` properties set to the center coordinate of the focus.
The `lat`, and `lon` properties get set to `46.63` and `11.22` degrees respectively.


Next, a set of functions are specified by adding the `functions` object to the JSON file.
These functions include `clip_layers`, `logify_scale`, and `adjust_focus`.

```json
	"functions": {
		"clip_datasets": {
			"clip": "hexbin+grids",
			"to": "regions"
		},
		"logify_scale": {
			"exp_type": "r"
		},
		"adjust_focus": false
	},
```

The `clip_layers` function is represented by an object containing the arguments to the function.
As the other examples have done, the `clip` and `to` arguments specify that the hexbin layer and grid layers be clipped to region layers.
Once again, the `logify_scale` function is represented by an object whose only member is the `exp_type` argument.
This specifies that the plot use a logarithmic scale with no exponents in the colour bar.
Next, since the focus has already been specified manually, and the function `adjust_focus` is performed by default, the function needs to be disabled.
To do this, the `adjust_focus` member of the `functions` object is set to `false`.


Finally, the output location of the visualization is specified in the JSON file through the `output` object.
The `output` object has two members `filepath`, and `crop_output` (set to true) which specify where the visualization is to be output, and that the output be cropped.

```json
	"output": {
		"filepath": "<path to output (.pdf)>",
		"crop_output": true
	}
}
```

% full JSON
The full JSON structure is given below.
The Python module translation of this JSON is given in `python_walkthrough.ipynb`, and `python_walkthrough.py`.

```json
{
	"hexbin_layer": {
		"data":"<path to data-1943.csv, data-1944.csv, data-1945.csv> (run for each year)",
		"latitude_field": "target_latitude",
		"longitude_field": "target_longitude",
		"hexbin_info": {
			"binning_field": "high_explosives_weight_tons",
			"binning_fn": "sum"
		},
		"hex_resolution":4,
		"manager": {
			"marker": {
				"line": {"width": 0.45}
			},
			"colorscale": "Viridis",
			"colorbar": {
				"x": 0.82
			}
		}
	},
	"regions": {
		"sample_Region_EUROPE": {
			"data": "EUROPE"
		}
	},
	"grids": {
		"sample_Grid_EUROPE": {
			"data": "EUROPE",
			"convex_simplify": true
		},
		"sample_Grid_RUSSIA": {
			"data": "RUSSIA"
		}
	},
	"grid_manager": {
		"marker": {
			"line": {"width": 0.45}
		}
	},
	"figure_geos": {
		"lataxis_range": [35, 58],
		"lonaxis_range": [0, 43],
		"projection_rotation": {
			"lat": 46.63321662159487,
			"lon": 11.21560455920799
		}
	},
	"functions": {
		"clip_datasets": {
			"clip": "hexbin+grids",
			"to": "regions"
		},
		"logify_scale": {
			"exp_type": "r"
		},
		"adjust_focus": false
	},
	"output": {
		"filepath": "<path to output (.pdf)>",
		"crop_output": true
	}
}
```