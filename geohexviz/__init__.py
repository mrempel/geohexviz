__version__ = "1.0.0"

import sys

try:
    import geopandas
except ModuleNotFoundError:
    print("The GeoHexViz library relies heavily on GeoPandas which is not easily installable via pip. Geopandas could "
          "not be found. If you haven't yet, ensure your dependencies were installed correctly.", file=sys.stderr)
    exit(1)