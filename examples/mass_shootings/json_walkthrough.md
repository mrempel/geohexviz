# Mass Shootings: GeoHexSimple JSON building walkthrough

The steps to building the JSON file for this visualization are very similar to the steps for the previous visualization.
The first step is the same; the data that is to be hexagonally binned, alongside it's configurations are passed into the `hexbin_layer` object in the JSON file.

```json
{
	"hexbin_layer": {
		"data": "data.csv",
		"hex_resolution":3,
		"hexbin_info": {
			"binning_field": "killed_injured",
			"binning_fn": "sum"
		},
		"manager": {
			"colorbar": {
				"x": 0.8365
			}
		}
	},
```

For this example, the `hexbin_layer` object has four members: `data`, `hex_resolution`, `hexbin_info`, and `manager`.
The `data` member is a full path to the data set containing the mass shooting locations.
The `hex_resolution` member controls the size of the hexagons and is now set to 3.
Unlike the first example, in this example the data is binned by the incident location and the value displayed is the sum of killed and injured in each hexagon.
To do this, the `hexbin_info` member is added.
It does this through its members `binning_field`, and `binning_fn`.
The `binning_field` member determines the grouped column to obtain the display value from, and the `binning_fn` member specifies how this display value is calculated.
In this example the `binning_field` is set to `killed_injured` which is a column in the data set containing the sum of killed and injured at each incident location.


For this example, the United States of America is to be highlighted as the region of interest.
To do this the `regions` object is added to the JSON file.

```json
	"regions": {
		"sample_Region_USA": {
			"data": "UNITED STATES OF AMERICA"
		}
	},
```

A single region is specified within the object and is referred to as `sample_Region_USA`.
The `data` member of `sample_Region_USA` is set to `UNITED STATES OF AMERICA`.


For this example, the epicenters of these incidents are to be displayed over the hexagonally binned data.
To do this the `points` object is added to the JSON file.

```json
	"points": {
		"sample_Point_EPICENTERS": {
			"data": "data-epicenters.csv",
			"text_field": "city",
			"manager": {
				"textposition": [
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
				"marker": {
					"symbol": "square-dot",
					"size": 4,
					"line": {
						"width": 0.5
					}
				}
			}
		}
	},
```

A single point layer is specified within the object and is referred to as `sample_Point_EPICENTERS`.
The `data` member of the `sample_Point_EPICENTERS` is set to a file containing the coordinates and names of the epicenters.
The `text_field` member of the `sample_Point_EPICENTERS` object is set to the name of the column containing the name of the epicenters.
This member controls the text to be displayed on top of each data entry on the map.
The `manager` member of `sample_Point_EPICENTERS` is an object that contains arguments that are passed to Plotly for this layer.
In this example the `manager` contains two members: `textposition`, and `marker`.
The `textposition` property controls the positioning of the text to be displayed alongside the scatter data.
In this case, since multiple epicenters are near each other, the positioning is set for each epicenter manually.
The `marker` member, which itself is an object, controls drawing properties for its associated layer.
In this example, the `marker` object is used to change the symbol used, the size of, and the outline width of each data point.
To change the symbol for each data point, the `symbol` member is added to the `marker` object, and set to `square-dot`.
To change the size of each data point, the `size` member is added to the `marker` object, and set to `4`.
Finally, to change the outline width for each data point, the `line` member, which is itself an object, is added `marker` object.
The `line` object controls various properties for the outline of each data point.
In this example, the width of the outline is set to `0.5` via the `width` member of the `line` object.


Next, a set of functions are specified within the JSON file under the `functions` object.

```json
	"functions": {
		"remove_empties": true,
		"adjust_focus": {
			"on": "hexbin",
			"buffer_lat": [0,15],
			"rot_buffer_lon": -8
		},
		"logify_scale": {
			"exp_type": "r"
		}
	},
```

The first member, `remove_empties` specifies that empty hexagons be removed from the data.
It is set to true as the function has no required arguments.
The second member, `adjust_focus` is an object specifying that the plot be focused on the data.
In this case, the `on` member of `adjust_focus` specifies that the plot be focused on the hexbin layer.
The `buffer_lat` member of `adjust_focus` specifies that the upper bound of the automatically calculated latitude range be shifted by 15 degrees.
The `rot_buffer_lon` member of `adjust_focus` specifies a number to add to the automatically calculated longitude rotation value.
For example, if the calculated rotation had a longitude of 8, and the `rot_buffer_lon` value was 2, then the final rotation longitude would be 10.
The final member, `logify_scale` is an object specifying that the plot use a logarithmic scale.
Once again, the `exp_type` member of `logify_scale` specifies that there be no exponent in the colour bar.


Finally, the output location of the visualization is specified in the JSON file through the `output` object.

```json
	"output": {
		"filepath": "output_visualization.pdf",
		"crop_output": true
	}
```

The full JSON structure is as follows.
The Python module translation of this JSON is listed in `python_walkthrough.ipynb`, and `python_walkthrough.py`.

```json
{
	"hexbin_layer": {
		"data": "data.csv",
		"hex_resolution":3,
		"hexbin_info": {
			"binning_field": "killed_injured",
			"binning_fn": "sum"
		},
		"manager": {
			"colorbar": {
				"x": 0.8365
			}
		}
	},
	"regions": {
		"sample_Region_USA": {
			"data": "UNITED STATES OF AMERICA"
		}
	},
	"points": {
		"sample_Point_EPICENTERS": {
			"data": "data-epicenters.csv",
			"text_field": "city",
			"manager": {
				"textposition": [
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
				"marker": {
					"symbol": "square-dot",
					"size": 4,
					"line": {
						"width": 0.5
					}
				}
			}
		}
	},
	"functions": {
		"remove_empties": true,
		"adjust_focus": {
			"on": "hexbin",
			"buffer_lat": [0,15],
			"rot_buffer_lon": -8
		},
		"logify_scale": {
			"exp_type": "r"
		}
	},
	"output": {
		"filepath": "output_visualization.pdf",
		"crop_output": true
	}
}
```