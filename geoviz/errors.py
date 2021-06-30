"""
Notes for refactoring:

We can perhaps integrate this with the builder class.
"""

class BuilderPlotBuildError(Exception):

    def __init__(self, message="An error occurred while plotting."):
        self.message = message
        super().__init__(self.message)


class BuilderDatasetInfoError(Exception):

    def __init__(self, message="Ivalid information was given with a dataset."):
        self.message = message
        super().__init__(self.message)

class BuilderAlterationError(Exception):

    def __init__(self, message="There was an error while altering data within the builder."):
        self.message = message
        super().__init__(self.message)

class ColorscaleError(Exception):

    def __init__(self, message="There was an error while reading the colorscale."):
        self.message = message
        super().__init__(self.message)

class DataFileReadError(Exception):

    def __init__(self, message="There was an error while reading the data file."):
        self.message = message
        super().__init__(self.message)

class DataReadError(Exception):

    def __init__(self, message="There was an error while reading the data."):
        self.message = message
        super().__init__(self.message)

class DataEmptyError(Exception):

    def __init__(self, message="The data can not be empty."):
        self.message = message
        super().__init__(self.message)

class BinValueTypeError(Exception):

    def __init__(self, message="The data can not be empty."):
        self.message = message
        super().__init__(self.message)