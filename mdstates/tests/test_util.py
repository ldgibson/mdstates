from os.path import abspath, dirname, join

from numpy.testing import assert_almost_equal

from ..util import getpath, Scaler


def test_getpath():

    data_dir_path = abspath(join(dirname(__file__), '..', 'data'))

    assert getpath('bond_distances.csv') ==\
        join(data_dir_path, 'bond_distances.csv'),\
        'Incorrect path provided.'

    return


def test_Scaler():
    def test_set_data_range():
        scaler = Scaler()
        scaler.set_data_range(0, 4)

        assert scaler.min_val == 0, "Mininum value incorrect."
        assert scaler.max_val == 0, "Maximum value incorrect."
        return

    def test_transform():
        scaler = Scaler()
        scaler.set_data_range(0, 4)

        assert_almost_equal(scaler.transform(0), 0.00,
                            err_msg="Bad transformation.")
        assert_almost_equal(scaler.transform(1), 0.25,
                            err_msg="Bad transformation.")
        assert_almost_equal(scaler.transform(2), 0.50,
                            err_msg="Bad transformation.")
        assert_almost_equal(scaler.transform(3), 0.75,
                            err_msg="Bad transformation.")
        assert_almost_equal(scaler.transform(4), 1.00,
                            err_msg="Bad transformation.")

        transformed = scaler.transform([0, 1, 2, 3, 4])
        correct_trans = [0.0, 0.25, 0.50, 0.75, 1.00]

        for trans, corr in zip(transformed, correct_trans):
            assert_almost_equal(trans, corr, err_msg="Bad transformation.")

        return
