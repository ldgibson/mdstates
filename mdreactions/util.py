from os.path import abspath, dirname, join

import pandas as pd


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
