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
