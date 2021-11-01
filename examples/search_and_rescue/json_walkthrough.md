# Search and Rescue: GeoHexSimple JSON building walkthrough

To make this visualization the required properties for the data to be hexagonally binned must be specified.
This is done by adding the hexbin_layer object to the JSON file.

```json
{
	"hexbin_layer": {
		"data":"<path to data.csv>",
		"hex_resolution":4,
		"manager": {
			"colorscale": "Viridis",
			"colorbar": {
				"x": 0.8325
			}
		}
	},
```

In this example, the `hexbin_layer` object has three members: `data`, `hex_resolution`, and `manager`.
The `data` member is a full path to the location of the data.
The `hex_resolution` member specifies the size of the hexagons to be used within the plot.
This number can range from 0 to 15 and is defined by **Uber H3**, where 0 represents the largest hexagon size, and 15 represents the smallest hexagon size.
Finally, the `manager` member, which itself is an object, specifies properties that are passed to Plotly.
In this case there are two being `colorscale`, and `colorbar`.
The `colorscale` member specifies the colour scale to be used within the plot.
By default, Plotly only allows the named colour scales to be continuous.
GeoHexViz overrides this behavior and allows all named colour scales available from Plotly.
The `colorbar` member is a collection of items that control different properties of the colour bar, such as background colour, border colour, and thickness.
In this example, the `x` value of the colour bar is being set which specifies the positioning of the colour bar; as the value goes from 0 to 1, the colour bar moves from left to right.

Next, since the region of Canada is to be highlighted, the `regions` object is added to the JSON file.

```json
	"regions": {
		"sample_Region_CANADA": {
			"data": "CANADA"
		}
	},
```

In this object there can be many defined regions, but for the sake of this visualization only one is needed.
This region is defined under the object `sample_Region_CANADA`, where the reference to the `data` defining the region is `CANADA`.
GeoHexViz recognizes the name of a country or continent as given by **Natural Earth Data** and automatically retrieves the geometries defining it.
Note that `sample_Region_CANADA` is the name that the layer will be referred to as, and could be something else.


Next, an extended grid layer is added to form a continuous grid.
This is done by adding the `grids` object to the JSON file.

```json
	"grids": {
		"sample_Grid_CANADA": {
			"data": "CANADA",
			"convex_simplify": true
		}
	},
```

Similar to the `regions` object, a grid referred to as `sample_Grid_CANADA` is specified, where its `data` member is also `CANADA`.
Due to that the H3 package supplies hexagons whose centroids are within the polygons, the polygon passed may not be completely filled with hexagons.
When set to true, the `convex_simplify` property attempts to fix this by expanding the polygon that was passed and then generating the grid.


Next, a set of functions are specified within the JSON file under the `functions` object.

```json
	"functions": {
		"clip_layers": {
			"clip": "hexbin+grids",
			 "to": "regions"
		},
		"adjust_focus": {
			"on": "regions",
			"buffer_lat": [0, 3]
		},
		"logify_scale": {
			"exp_type": "r"
		}
	},
```

The first object `clip_layers` specifies the data is to be clipped only to the region of Canada.
Specifically it does this through the members in the object; the `clip` member specifies what layers to clip and the `to` member specifies the layers to act as the boundary of the clip.
In this case, the `clip` member is `hexbin+grids` which refers to the hexbin layer and any grid layers present.
The `to` member is `regions` which refers to any region layers present.
The second object `adjust_focus` specifies that the plot be focused on the region of Canada.
The `on` member specifies which layers to focus on; in this case it is specified to `regions` which refers to any region layers present.
The `buffer_lat` member specifies two numbers that will be added to the lower and upper values of the automatically calculated boundary, 
i.e., if the automatically calculated latitude range was from 0 to 50, with a `buffer_lat` member of [10, 20], the resulting latitude range would be from 10 to 70.
The final object `logify_scale` specifies that the plot use a logarithmic scale (using raw text).
The `exp_type` member specifies what type of exponent is to be used in the colour bar; the value `r` means that the raw numbers will be displayed on the colour bar, i.e., 1, 10, 100, 1000, etc.
The possible properties for each function are described in the official documentation.


Finally, the output location of the visualization is specified in the JSON file through the `output` object.

```json
	"output": {
		"filepath": "<path to output (.pdf)>",
		"crop_output": true
	}
}
```

The first member, `filepath`, specifies the destination of output visualization; the extension of the file path determines the file type.
The second member `crop_output` specifies that the output visualization be cropped via **PdfCropMargins**.
When set to true, the `crop_output` member requires that the user have **PdfCropMargins**, alongside its dependencies installed in their environment.

The complete JSON file is given below.
The Python module translation of this JSON structure is given in this directory under `python_walkthrough.ipynb`, and `python_walkthrough.py`.

```json
{
	"hexbin_layer": {
		"data":"<path to data.csv>",
		"hex_resolution":4,
		"manager": {
			"colorscale": "Viridis",
			"colorbar": {
				"x": 0.8325
			}
		}
	},
	"regions": {
		"sample_Region_CANADA": {
			"data": "CANADA"
		}
	},
	"grids": {
		"sample_Grid_CANADA": {
			"data": "CANADA",
			"convex_simplify": true
		}
	},
	"functions": {
		"clip_layers": {
			"clip": "hexbin+grids",
			 "to": "regions"
		},
		"adjust_focus": {
			"on": "regions",
			"buffer_lat": [0, 3]
		},
		"logify_scale": {
			"exp_type": "r"
		}
	},
	"output": {
		"filepath": "<path to output (.pdf)>",
		"crop_output": true
	}
}
```