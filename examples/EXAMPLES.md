# Welcome to the Example Usage for the GeoHexViz package!

The general overview of this directory is listed below:

| Path                        | Description                                                                                                         |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------|
| examples-dependencies       | Dependencies for the tutorials and examples to follow                                                               |
| tutorial-geohexsimple.ipynb | Tutorial: information for the usage of GeoHexSimple                                                                 |
| tutorial-geohexviz.ipynb    | Tutorial: information for the usage of GeoHexViz                                                                    |
| tutorial-installation.md    | Tutorial: the general process of installation and other information                                                 |
| search_and_rescue           | Example: density of search and recue incidents over the Canadian landmass (randomly generated locations)            |
| mass_shootings              | Example: killed or injured during mass shootings in the United States of America (late 2017 to late 2021) [[1]](#1) |
| ww2_bombings                | Example: weight (in tons) of bombs dropped in World War 2 (1943 to 1945) [[2]](#2)                                  |
| forest_fires                | Example: most frequent category of forest fires in the United States of America (2017) [[3]](#3)                    |

This directory provides useful information and tutorials pertaining to the usage of GeoHexViz, and GeoHexSimple.
This initial tutorials such as:
1. information for the usage of GeoHexSimple (tutorial-geohexsimple.ipynb);
2. information for the usage of GeoHexViz (tutorial-geohexviz.ipynb); and   
3. the general process of installation (tutorial-installation.md).

This directory also contains four examples depicting example usage for GeoHexViz.
The four examples pertain to:
1. density of search and recue incidents over the Canadian landmass (randomly generated locations);
2. killed or injured during mass shootings in the United States of America (late 2017 to late 2021) [[1]](#1);
3. weight (in tons) of bombs dropped in World War 2 (1943 to 1945) [[2]](#2); and
4. most frequent category of forest fires in the United States of America (2017) [[3]](#3).
These four examples are also described in detail within the reference document published alongside this package.
   

As previously mentioned, there are two methods to creating visualizations.
Method 1 concerns using the GeoHexSimple package's script to run a file containing plot structure.
Method 2 concerns using Python code to interact with the functions within the package.
Method 2 method has two categories:

a) Using functions from the GeoHexSimple package \
b) Using functions from the GeoHexViz package

These methods are described in-detail within the reference document published alongside this package,
as well as in the README file for this project.

Each of the example directories mentioned above provide:
* the resource file(s) required for creating the visualization (.csv);
* a Markdown file containing information about the example and its contents (.md);
* a JSON file containing the required structure for Method 1 (.json);
* a Markdown file containing a walkthrough on how to build the JSON file for Method 1 (.md);
* a PDF file containing the output visualization (.pdf);
* a Jupyter Notebook file containing a walkthrough on how to build the visualization for Method 2 (.ipynb); and
* a Python file containing a walkthrough on how to build the visualization for Method 2 (.py).



## Data Source References
<a id="1">[1]</a> 
GVA (2021), Gun Violence Archive (online), gunviolencearchive, https://www.gunviolencearchive.org/reports/mass-shooting (Access Date: October 2021).

<a id="1">[2]</a> 
Larion, A. (2016), Aerial Bombing Operations in World War II (online), Kaggle, https://www.kaggle.com/usaf/world-war-ii?select=operations.csv (Access Date: September 2021)

<a id="1">[3]</a> 
MBTS (2021), Monitoring Trends in Burn Severity Burned Area Boundaries (Feature Layer) (online), ArcGIS, https://hub.arcgis.com/datasets/usfs::monitoring-trends-in-burn-severity-burned-area-boundaries-feature-layer/about (Access Date: September 2021)