"""
Notes for refactoring:

We can perhaps integrate this with the builder class.
"""


class BuilderPlotBuildError(Exception):

    def __init__(self, message: str = "An error occurred while plotting."):
        self.message = message
        super().__init__(self.message)


class BuilderDatasetInfoError(Exception):

    def __init__(self, message: str = "Ivalid information was given with a dataset."):
        self.message = message
        super().__init__(self.message)


class BuilderAlterationError(Exception):

    def __init__(self, message: str = "There was an error while altering data within the builder."):
        self.message = message
        super().__init__(self.message)


class ColorscaleError(Exception):

    def __init__(self, message: str = "There was an error while reading the colorscale."):
        self.message = message
        super().__init__(self.message)


class DataFileReadError(Exception):

    def __init__(self, message: str = "There was an error while reading the data file."):
        self.message = message
        super().__init__(self.message)


class DataReadError(Exception):

    def __init__(self, message: str = "There was an error while reading the data."):
        self.message = message
        super().__init__(self.message)


class DataEmptyError(Exception):

    def __init__(self, message: str = "The data can not be empty."):
        self.message = message
        super().__init__(self.message)


class BinValueTypeError(Exception):

    def __init__(self, message: str = "The data can not be empty."):
        self.message = message
        super().__init__(self.message)


class DatasetNamingError(Exception):

    def __init__(self, name_error: str):
        self.message = f'The name that was given to the dataset is invalid. Error type: {name_error}.'
        super().__init__(self.message)


class MainDatasetNotFoundError(Exception):

    def __init__(self):
        self.message = 'The main dataset could not be found.'
        super().__init__(self.message)


class DatasetNotFoundError(Exception):

    def __init__(self, name: str, dstype: str):
        self.message = f"The {dstype} type dataset named '{name}' could not be found."
        super().__init__(self.message)


class BuilderQueryInvalidError(Exception):

    def __init__(self, message: str = "The input query was invalid."):
        self.message = message
        super().__init__(self.message)


class NoDatasetsError(Exception):

    def __init__(self, message: str = "There were no datasets found."):
        self.message = message
        super().__init__(self.message)
