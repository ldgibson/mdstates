from os.path import abspath, dirname, join

from ..util import getpath


def test_getpath():

    data_dir_path = abspath(join(dirname(__file__), '..', 'data'))

    assert getpath('bond_distances.csv') ==\
        join(data_dir_path, 'bond_distances.csv'),\
        'Incorrect path provided.'

    return
