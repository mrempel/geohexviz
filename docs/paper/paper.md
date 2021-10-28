---
title: 'GeoHexViz: A Python package for the visualization of hexagonally binned geospatial data'
tags:
  - Python
  - geospatial
  - hexagon
  - binning
  - operations research
authors:
  - name: Tony M. Abou Zeidan^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0001-5130-3070
  - name: Mark Rempel^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-6248-1722
date: 25 October 2021
bibliography: paper.bib
---

*[JSON]: JavaScript Object Notation

# Summary

Geospatial visualization is an important communication method that is often used in military operations research to convey analyses to both analysts and decision makers. For example: to help commanders coordinate units over a large geographic region [@feibush2000a], to depict how different terrain impacts the performance of vehicles [@laskey2010a]; and to inform training decisions in order to meet mission requirements [@goodrich2019a]. When these analyses include a large amount of point-like data, binning---in particular, hexagonal binning---may be used to summarize the data and subsequently produce an effective visualization, such as communicating risk military operations [@hunter2021a] or in the context of public safety and COVID-19 [@shaito2021a]. However, creating such visualizations may be difficult for many since it requires in-depth knowledge of both Geographic Information Systems and analytical techniques, not to mention access to software that may require a paid license, training, and perhaps knowledge of a programming language. Open source software that provides a simple interface, requires minimal in-depth knowledge, and either limited or no programming will allow a wider range of operations researchers to produce high-quality visualizations---ultimately, leading to better informed decisions.

GeoHexViz is accessible at \textbf{FILL ME IN} and is installed via a `setup.py` script.

# Statement of need

Creating geospatial visualizations is often time-consuming and laborious [@vartak2014a]. This is due to that in-depth knowledge of Geographic Information System (GIS) concepts is required to use GIS software, and hence to create geospatial visualizations [@bertazzon2013a]. For example, an individual must decide which map projection to use, the colour scheme, the basemap, and in some cases how to organize the data in layers. There are many software applications that may be used to create geospatial visualizations, such as ArcGIS [@dangermond2021a], QGIS [@sherman2021a], and D3 [@bostock2021a]. ArcGIS provides a wide range of capabilities, but requires a paid license and a solid foundation in geospatial information processing [@arcgis-limits]. In contrast, QGIS is free and open-source, but also requires an in-depth knowledge of geospatial information processing to be used effectively [@qgis-limits]. As an alternative, programming-based approaches to create visualizations, such as D3 [@bostock2021a] and Plotly [@plotly-web], have been developed in the last decade. In addition to an understanding of geospatial concepts, they also requires a knowledge
of JavaScript and Python respectively. Common across these applications is the requirement to have knowledge of geospatial concepts, and acquiring this knowledge has been identified as a significant challenge [@sipe2003a].

In addition to being time consuming to create, geospatial visualizations often require analysts to have specialized knowledge of analytic techniques. One of these techniques is binning in which a grid is placed over a data set and the individual data points are grouped by grid cell [@binning_info]. This method is used when it is difficult to visualize geospatial point-like data sets; in particular, when the number of points is large, they become difficult to distinguish [@binning2, @binning_info]. In order to provide an accurate representation of the binned data, an analyst must choose a versatile grid type. There are many grid types available, such as circular, rectangular, and hexagonal. A circular grid is optimal for analysis purposes since circles are accurate for sampling, but does not provide a continuous grid [@sinha2019a]. Rectangular grids are simple to implement; however, may not be suitable when investigating connectivity or movement [@rectangles]. A hexagonal grid is often selected because its more visually appealing than other grid types [@battersby2017a], and shares many of its properties with a circular grid [@sinha2019a]. In addition, hexagonal grids oer many advantages including: hexagons have the same number of neighbours as they does edges; the center of each hexagon is equidistant from the centers of its neighbours (which helps when analyzing connectivity or movement); and hexagons tile densely on curved surfaces, resulting in lower edge effects (reducing analytic bias) [@sinha2019a]. The previously mentioned GIS systems provide functionality to perform hexagonal binning, albeit access to this functionality is often limited due to the issues described above.

With this in mind, GeoHexViz aims to reduce the time and in-depth knowledge required to produce publication-quality geospatial visualizations that use hexagonal binning. GeoHexViz, which is built on top of several underlying Python packages, allows an analyst to produce a publication-quality visualization in two ways. First, a user may generate a visualization via running a pre-existing command-line script whose input is a single JSON le that defines the properties of the visualization. Second, a user may generate a visualization by writing a Python script that imports and invokes functions on objects found in the GeoHexViz's Python modules. Both methods require that the user provide only two arguments. The first argument is a reference to the data, which is a file path or may be a DataFrame [@mckinney2021a] or GeoDataFrame [@jordahl2021a}] when using the second option. The second argument is a reference to the columns within the data that define the latitudes, longitudes, and the value associated with each. If no value column is present, the default of each data entry is set to one.

# Features

The aim of GeoHexViz is to simplify the production of publication-quality geospatial visualizations that utilize hexagonal binning. To do this, a user specifies a set of *layers*---where each layer is defined as a ``[group] of point, line, or area (polygon) features representing a particular class or type of real-world entities'' [@layer-definition]---to be visualized. At a minimum, the user must specify one layer, the *hexbin layer*, through two arguments\textemdash a reference to the point-like data to be hexagonally binned, and references to the columns containing latitudes, longitudes, and value at each coordinate. 

If the output visualization is not satisfactory, GeoHexViz allows a user to adjust features of the plot. These features include:

+ **scale**: the data displayed in the visualization may be on a linear (default) or logarithmic scale;
+ **colour scale**: the colour scale of the visualization may be continuous (default) or discrete;
+ **focus**: the visualization may have no focal point (default), showing a view of the whole Earth, or may be focused on the data; and
+ **filtering**: all of the data may be present in the visualization (default) or may be clipped to a geographic region.

In addition, a user can change other properties of the visualization, such as border colour, land colour, sea colour, and figure size. In this case, these properties are passed by GeoHexViz to Plotly.

GeoHexViz may be used to create a visualization in two ways. First, the user can use GeoHexViz's command-line script `GeoHexSimple` to read a JSON file that contains properties for the visualization.
The purpose of the command-line script is to give non-technical users a simple interface.
Second, the user can generate a visualization via importing and invoking functions found in the GeoHexViz's Python modules. When using the second method, the reference to the data may be a DataFrame [@pandas-dataframe-docs] or GeoDataFrame [@geopandas-geodataframe-docs] object. If the input reference is a GeoDataFrame, the package does not *need* latitude or longitude columns. Rather, the input to the software will be the entries within the geometry column (it is up to the user to ensure that these are valid geometry types).

The hexagonal tiling is generated with the use of the Uber H3 library [@uber-H3], which provides an interface to convert conventional lat/long coordinates into a geospatial index. These hex-tiles are then stored within GeoPandas objects and are binned by common hex tile [@geopandas]. The data is then converted into GeoJSON format and is visualized with the aid of the Plotly graphing library [@plotly-py].

The resulting output is a publication-quality visualization, that can be displayed, or output to a file.
GeoHexViz can generate visualizations for both quantitative, and qualitative data sets.s
![Bombings in World War 2\textemdash European Theatre; Total mass of bombs dropped in tons (1943)\label{fig:examplequant-ww2-subA}](bombings-1943.pdf){#ww2A width=30%}
![Bombings in World War 2\textemdash European Theatre; Total mass of bombs dropped in tons (1944)\label{fig:examplequant-ww2-subB}](bombings-1944.pdf){#ww2B width=same}
![Bombings in World War 2\textemdash European Theatre; Total mass of bombs dropped in tons (1945)\label{fig:examplequant-ww2-subC}](bombings-1945.pdf){#ww2C width=same}
![Most frequent fire category by location (United States of America: 2017)\label{fig:examplequal-fires}](fire_locations.pdf)

## Saving the output

Given a JSON file that defines the hexbin layer and optional layers, such as regions, grids, and outlines, GeoHexViz outputs a geospatial visualization with hexagonally binned data. The visualization may be saved in a variety of formats, including PDF, PNG, JPEG, WEBP, SVG, and EPS formats.

# Limitations
This package uses GeoJSON format to plot data sets. With GeoJSON comes difficulties when geometries cross the 180th meridian [@meridian]. The issue appears to cause a color that bleeds through the entire plot and leaves a hexagon empty. In the final plot, this issue may or may not appear as it only occurs at certain angles of rotation. In this package a simple solution to the problem is implemented, in the future it would be best to provide a more robust solution. The solution that is used works generally, however, when hexagons containing either the north or south pole are present, the solution to the 180th meridian issue persists. This pole issue can be seen in \autoref{fig:sar-issue}.

There also exists some issues with the generation of discrete color scales under rare circumstances. These circumstances include generating discrete color scales with not enough hues to fill the scale, and generating diverging discrete colorscales with the center hue in a weird position. These issues have been noted and will be fixed in the near future.

There exists issues with the positioning and height of the color bar with respect to the plot area of the figure. Although the user is capable of altering the dimensions and positioning of the color bar, this should be done automatically as it is a common feature of publication quality choropleth maps.

# Acknowledgements

Thank you to Nicholi Shiell for his input in testing, and providing advice for the development of this package and of its supporting documents.

# References