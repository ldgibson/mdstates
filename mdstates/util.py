import json
import os.path
from os.path import abspath, dirname, join
from numbers import Number

import numpy as np
import pandas as pd


def json_to_string(json_dict):
    return json.dumps(json_dict)


def json_to_file(json_dict, filename):
    with open(filename, 'w') as f:
        json.dump(json_dict, f)
    return


def load_json_from_file(filename):
    if os.path.exists(filename):
        pass
    else:
        raise FileNotFoundError("File not found: {}".format(filename))

    with open(filename, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def load_json_from_string(json_string):
    json_dict = json.loads(json_string)
    return json_dict


def getpath(filename):
    """
    Returns the absolute path of a file in the data directory.
    """
    return abspath(join(dirname(__file__), "data", filename))


def loadfile(filename, **kwargs):
    """
    Loads the file at the given path into a pandas dataframe
    and returns it.
    """
    return pd.read_csv(getpath(filename), **kwargs)


def find_nearest(n, arr):
    """Finds nearest value in array.

    Parameters
    ----------
    n : int or float
    arr : array-like container of int or float

    Returns
    -------
    int or float"""

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    else:
        pass

    idx = (np.abs(arr - n)).argmin()
    return arr[idx]


class Scaler:
    def __init__(self, target_min=0, target_max=1):
        self.target_min = target_min
        self.target_max = target_max
        self.min_val = None
        self.max_val = None
        return

    def set_data_range(self, min_val, max_val):
        """Sets the mininum and maximum range of working data.

        Parameters
        ----------
        min_val, max_val : float or int
        """
        assert min_val <= max_val,\
            "First argument must be less or equal to than second argument."
        self.max_val = max_val
        self.min_val = min_val
        return

    def transform(self, data):
        """Transforms a datum or list of data into the normalized scale.

        Parameters
        ----------
        data : float or int or list of float or int

        Returns
        -------
        float or list of float
            Transformed value into the user specified scale."""
        if isinstance(data, Number):

            if data > self.max_val or data < self.min_val:
                raise ValueError("{} is out of range.".format(data))
            elif self.max_val == self.min_val:
                return self.max_val
            else:
                return (data - self.min_val) / (self.max_val - self.min_val) *\
                    (self.target_max - self.target_min) + self.target_min
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            for d in data:
                if d > self.max_val or d < self.min_val:
                    raise ValueError("{} is out of range.".format(d))
            if self.max_val == self.min_val:
                return [self.max_val for _ in data]
            else:
                return [(d - self.min_val) / (self.max_val - self.min_val) *
                        (self.target_max - self.target_min) + self.target_min
                        for d in data]
