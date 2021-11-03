from enum import Enum


class LayerType(Enum):
    """An enumeration of different layer types.
    """
    HEXBIN = 'hexbin'
    REGION = 'region'
    OUTLINE = 'outline'
    GRID = 'grid'
    POINT = 'point'


class BuilderPlotBuildError(Exception):

    def __init__(self, message: str = "An error occurred while plotting."):
        self.message = message
        super().__init__(self.message)


class BuilderAlterationError(Exception):

    def __init__(self, message: str = "There was an error while altering data within the builder."):
        self.message = message
        super().__init__(self.message)


class ColorScaleError(Exception):

    def __init__(self, message: str = "There was an error while reading the colorscale."):
        self.message = message
        super().__init__(self.message)


class DataFileReadError(Exception):

    def __init__(self, name: str, dstype: LayerType, message: str = "There was an error while reading the data file."):
        if dstype == LayerType.HEXBIN:
            self.message = f"There was an error while reading the 'data' property of the {dstype.value} layer.\n" \
                           f"Error message:\n{message}"
        else:
            self.message = f"There was an error while reading the 'data' property of the " \
                           f"{dstype.value}-type layer named {name}.\n" \
                           f"Error message:\n{message}"
        super().__init__(self.message)


class DataReadError(Exception):

    def __init__(self, name: str, dstype: LayerType,
                 message: str = "There was an error while reading the 'data' parameter passed."):
        if dstype == LayerType.HEXBIN:
            self.message = f"There was an error while reading the 'data' property of the {dstype.value} layer.\n" \
                           f"Error message:\n{message}"
        else:
            self.message = f"There was an error while reading the 'data' property of the " \
                           f"{dstype.value}-type layer named {name}.\n" \
                           f"Error message:\n{message}"
        super().__init__(self.message)


class DataTypeError(Exception):

    def __init__(self, name: str, dstype: LayerType, allow_builtin: bool):

        self.options = ["file path", "DataFrame", "GeoDataFrame"]
        if allow_builtin:
            self.options.insert(0, "continent name")
            self.options.insert(0, "country name")

        self.name = name
        self.dstype = dstype
        self.allow_builtin = allow_builtin
        if dstype == LayerType.HEXBIN:
            self.message = f"The 'data' parameter given for the {dstype.value} layer was invalid.\n" \
                           f"The data must be one of the following: {', '.join(self.options)}"
        else:
            self.message = f"The 'data' parameter given for the {dstype.value}-type layer named {name} " \
                           f"was invalid.\nThe data must be one of the following: {', '.join(self.options)}"

        super().__init__(self.message)


latitude_aliases = ["latitude", "latitudes", "lat", "lats", "latitude_field"]
longitude_aliases = ["longitude", "longitudes", "lon", "lons", "long", "longs", "longitude_field"]


class GeometryParseLatLongError(Exception):

    def __init__(self, name: str, dstype: LayerType, lat: bool):

        self.name = name
        self.dstype = dstype
        self.fmt = "latitude" if lat else "longitude"
        if dstype == LayerType.HEXBIN:
            self.message = f"There was no geometry passed for the {dstype} layer.\n" \
                           f"In these cases, the columns containing latitudes " \
                           f"and longitudes must be specified (missing {self.fmt});\n" \
                           f"unless the naming of these columns follow builtin naming conventions.\n" \
                           f"Valid latitude column names: {latitude_aliases}," \
                           f"\nValid longitude column names: {longitude_aliases}"
        else:
            self.message = f"There was no geometry passed for the {dstype}-type layer named {name}.\n" \
                           f"In these cases, the columns containing latitudes " \
                           f"and longitudes must be specified (missing {self.fmt});\n" \
                           f"unless the naming of these columns follow builtin naming conventions.\n" \
                           f"Valid latitude column names: {latitude_aliases}," \
                           f"\nValid longitude column names: {longitude_aliases}"

        super().__init__(self.message)


class LatLongParseTypeError(Exception):

    def __init__(self, name: str, dstype: LayerType, lat: bool):

        self.name = name
        self.dstype = dstype
        self.fmt = "latitude" if lat else "longitude"
        if dstype == LayerType.HEXBIN:
            self.message = f"A {self.fmt} column was passed or parsed for the {dstype} layer.\n" \
                           f"The column does not contain numeric entries, " \
                           f"and could not be converted to numerical format."
        else:
            self.message = f"A {self.fmt} column was passed or parsed for the " \
                           f"{dstype}-type layer named {name}.\n" \
                           f"The column does not contain numeric entries, " \
                           f"and could not be converted to numerical format."

        super().__init__(self.message)


class DataEmptyError(Exception):

    def __init__(self, message: str = "The data can not be empty."):
        self.message = message
        super().__init__(self.message)


class BinValueTypeError(Exception):

    def __init__(self, message: str = "The data can not be empty."):
        self.message = message
        super().__init__(self.message)


class LayerNamingError(Exception):

    def __init__(self, name: str, dstype: LayerType, err_type: str):
        self.name = name
        self.dstype = dstype
        self.message = f"An invalid name was passed when adding a {dstype}-type layer." \
                       f"\nName: {name}\nInvalidity: {err_type}"
        super().__init__(self.message)


class NoLayerError(Exception):

    def __init__(self, name: str, dstype: LayerType):
        self.dstype = dstype
        if dstype == LayerType.HEXBIN:
            self.message = f"There is no {dstype.value} layer present."
        else:
            self.message = f"There is no {dstype.value}-type layer named '{name}' present."
        super().__init__(self.message)


class NoLayersError(Exception):

    def __init__(self, dstype: LayerType):
        self.dstype = dstype
        self.message = f"There are no {dstype.value}-type layers present."
        super().__init__(self.message)


class NoHexagonalTilingError(Exception):

    def __init__(self, name: str, dstype: LayerType):

        self.name = name
        self.dstype = dstype
        if dstype == LayerType.HEXBIN:
            self.message = f"No hexagonal tiling could be generated for the {dstype.value} layer.\n"
        else:
            self.message = f"No hexagonal tiling could be generated for the" \
                           f" {dstype.value}-type layer named {name}.\n"
        super().__init__(self.message)


class _BuilderQueryInvalidError(Exception):

    def __init__(self, message: str = "The input query was invalid."):
        self.message = message
        super().__init__(self.message)


class BigQueryError(Exception):

    def __init__(self, problematic: str):
        self.problematic = problematic
        self.message = f"The given query contains an argument that refers to a collection of layers.\n" \
                       f"Problematic: {problematic}"


class BuilderQueryInvalidError(Exception):

    def __init__(self, message: str = "The input query was invalid."):
        self.message = message
        super().__init__(self.message)


class NoFilepathError(Exception):

    def __init__(self, message: str = "There was no filepath provided."):
        self.message = message
        super().__init__(self.message)
