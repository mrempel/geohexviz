from pandas import DataFrame
from copy import deepcopy

from pandas.core.dtypes.inference import is_hashable


class DataSet:

    def __init__(self, data: DataFrame, **kwargs):
        self.data = data
        self.others = kwargs

    def isvalid(self, validothers=None):
        if validothers is None:
            validothers = []
        return not self.data.empty and all(x in self.others for x in validothers)

    def __iter__(self):
        return iter({
            'data': self.data,
            **self.others
        })

    def __copy__(self):
        return DataSet(data=self.data.copy(), **self.others)

    def __deepcopy__(self, memodict={}):
        return DataSet(data=self.data.copy(deep=True), **deepcopy(self.others))


from collections.abc import MutableMapping


class DataSetDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, data: DataFrame, **kwargs):
        self.data = data
        self.store = dict()
        self.update(dict(**kwargs))  # use the free update to set keys

    def __getitem__(self, item):
        if item == 'data':
            return self.data
        else:
            return self.store[item]

    def __setitem__(self, key, value):
        if key == 'data':
            self.data = value
        else:
            self.store[key] = value

    def __delitem__(self, key):
        if key == 'data':
            raise AttributeError("Cannot delete data entry.")
        del self.store[key]

    def keytransform(self):
        to_pop = []
        for k, v in self.store.items():
            if is_hashable(v) and v in self.data.columns:
                self.data.rename({v: k}, axis=1, inplace=True)
                to_pop.append(k)
        for p in to_pop:
            self.store.pop(p)

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.data)

    def __copy__(self):
        return DataSetDict(self.data.copy(), **self.store.copy())

    def __deepcopy__(self, memodict={}):
        return DataSetDict(self.data.copy(deep=True), **deepcopy(self.store))