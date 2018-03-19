import mdtraj as md
import numpy as np

from .. import core

xyz_file = '/Users/ldgibson/Development/mdreactions/' +\
           'mdreactions/tests/test_cases/test_case.xyz'
top_file = '/Users/ldgibson/Development/mdreactions/' +\
           'mdreactions/tests/test_cases/test_top.pdb'


def test_loadtraj():
    # Ensure an mdtraj.Trajectory object is returned
    traj = core.loadtraj(xyz_file, top_file)
    assert isinstance(traj, md.Trajectory), \
        '"traj" is not of type: mdtraj.Trajectory'

    # Test assertion for trajectory file existence
    try:
        core.loadtraj('bad_traj_file', top_file)
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion',
                        'Bad trajectory file was accepted without error')

    # Test assertion for topology file existence
    try:
        core.loadtraj(xyz_file, 'bad_top_file')
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion',
                        'Bad topology file was accepted without error')
    return


def test_generate_pairs():

    # Ensure correct number of pairs is calculated
    test_num = 5
    true_pairs = test_num * (test_num - 1) / 2
    test_pairs = core.generate_pairs(test_num)

    assert np.shape(test_pairs) == (true_pairs, 2), \
        'Bad shape to list of pairs: ' + str(np.shape(test_pairs))

    # Test assertion for TypeError
    try:
        core.generate_pairs('ten')
    except (TypeError):
        pass
    else:
        raise Exception('Failed TypeError', 'The value passed must be numeric')

    # Check if incorrect pairs are being generated
    # - Check for pairs that are of an atom with itself
    self_pairs = []
    for i in range(test_num):
        self_pairs.append([i, i])

    for pair in self_pairs:
        if pair in test_pairs:
            raise Exception('Bad pair',
                            'Generating pairs of an atom and itself')
        else:
            pass

    # Check for pairs that have the second atom index lower than the
    # first atom index
    bad_pairs = []
    for i in range(test_num-1):
        for j in range(1, test_num):
            if i < j:
                bad_pairs.append([j, i])
            else:
                pass

    for pair in bad_pairs:
        if pair in test_pairs:
            raise Exception('Bad pair',
                            'First atom index should always ' +
                            'be larger than the second')
        else:
            pass

    return


def test_compute_distances():

    test_traj = md.load(xyz_file, top=top_file)

    NUM_ATOMS = test_traj.n_atoms
    NUM_PAIRS = int(NUM_ATOMS * (NUM_ATOMS - 1) / 2)
    pairs = np.zeros((NUM_PAIRS, 2))
    idx = 0

    for i in range(0, NUM_ATOMS-1):
        for j in range(i+1, NUM_ATOMS):
            pairs[idx, :] = [i, j]
            idx = idx + 1

    distances = core.compute_distances(test_traj, pairs, periodic=False)

    assert distances.shape == (test_traj.n_frames, NUM_PAIRS), \
        "Shape of `distances' array is incorrect."
    return


def test_build_connections():
    # test distance matrix
    dist = np.array([[2.000, 4.000], [6.000, 8.000]])

    # Test negative cutoff value error
    try:
        core.build_connections(dist, CUTOFF=-0.170)
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion error',
                        'Negative cutoff value was allowed.')

    # Test 0 cutoff value error
    try:
        core.build_connections(dist, CUTOFF=0)
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion error',
                        'Cutoff value of 0 is not allowed.')
    return


def test_linear_to_square():
    rxn_matrix = np.array([[0, 1, 0],
                           [1, 0, 1]], dtype=int)

    real_cmat = np.array([[[0, 0, 1],
                           [0, 0, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]]])

    cmat = core.linear_to_square(rxn_matrix)

    # Check that there are no nonzero values in the lower triangle
    assert np.count_nonzero(np.tril(cmat)) == 0, \
        'Contact matrix not being structured properly. ' + \
        'Nonzero values found in lower triangle of matrix.'

    # Check that the correct contact matrix is being built
    assert (cmat == real_cmat).all(), \
        'Failure to build correct contact matrix.'

    return


def test_generate_ignore_list():
    cmat = np.zeros((10, 3, 3))

    # Contact matrix before i = 4:
    # [[0 1 0],
    #  [0 0 1],
    #  [0 0 0]]
    #
    # Contact matrix after i = 4:
    # [[0 1 0],
    #  [0 0 0],
    #  [0 0 0]]
    cmat[:, 0, 1] = 1
    cmat[:, 1, 2] = 1
    cmat[4, 1, 2] = 0

    real_ignore_list = [[[0, 2]],
                        [[0, 1]]]

    ignore_list = core.generate_ignore_list(cmat)

    assert ignore_list[0] == real_ignore_list[0], \
        'Incorrect unbonded ignore_list being generated'

    assert ignore_list[1] == real_ignore_list[1], \
        'Incorrect bonded ignore_list being generated'

    return


def test_viterbi():
    assert False, "No unit tests written yet"
    return


def test_find_reaction_frames():

    testmat = np.array([[[0, 1],
                       [0, 0]],
                      [[0, 1],
                       [0, 0]],
                      [[0, 0],
                       [0, 0]]])

    test_frames = core.find_reaction_frames(testmat)

    assert test_frames == (2,), 'Incorrect reaction frames found'

    return


def test_generate_SMILES():
    assert False, "No unit tests written yet"

    return
