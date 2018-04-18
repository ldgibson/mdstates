import os.path

import mdtraj as md
import numpy as np

from ..main import Network

currentdir = os.path.dirname(__file__)
testdir = os.path.join(currentdir, 'test_cases')
traj_path = os.path.join(testdir, 'test_case.xyz')
top_path = os.path.join(testdir, 'test_top.pdb')


def test_add_replica():
    net = Network()
    net.add_replica(traj_path, top_path)
    assert isinstance(net.replica[0]['traj'], md.Trajectory), \
        '"traj" is not of type: mdtraj.Trajectory'

    # Test assertion for trajectory file existence
    try:
        net.add_replica('bad_traj_file', top_path)
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion',
                        'Bad trajectory file was accepted without error')

    # Test assertion for topology file existence
    try:
        net.add_replica(traj_path, 'bad_top_file')
    except (Exception):
        pass
    else:
        raise Exception('Failed assertion',
                        'Bad topology file was accepted without error')
    return


def test_remove_replica():
    net = Network()
    net.add_replica(traj_path, top_path)
    net.remove_replica(0)

    assert not net.replica, "Replica should be emtpy."

    try:
        net.remove_replica(0)
    except(IndexError):
        pass
    else:
        raise Exception("Bad index allowed.")

    return


def test_generate_contact_matrix():
    net = Network()
    net.add_replica(traj_path, top_path)
    net.generate_contact_matrix()

    assert np.all([x == 0 or x == 1 for x in
                   np.nditer(net.replica[0]['cmat'])])

    assert (net.replica[0]['cmat'][:, 0, 1] == 1).all() and\
           (net.replica[0]['cmat'][:, 0, 2] == 1).all() and\
           (net.replica[0]['cmat'][:, 0, 3] == 1).all()

    n_frames = net.replica[0]['traj'].n_frames

    assert net.replica[0]['cmat'].shape ==\
        (n_frames, net.n_atoms, net.n_atoms)

    return


def test__generate_pairs():
    net = Network()
    net.add_replica(traj_path, top_path)

    net._generate_pairs()

    pairs = np.triu_indices(6, 1)

    for i, j in zip(*pairs):
        assert [i, j] in net._pairs

    return


def test_set_cutoff():
    net = Network()
    net.add_replica(traj_path, top_path)

    try:
        net.set_cutoff('CC', 0.2)
    except(TypeError):
        pass
    else:
        raise Exception("Non-list allowed.")

    try:
        net.set_cutoff(['C'], 0.2)
    except(ValueError):
        pass
    else:
        raise Exception("Single element allowed.")

    try:
        net.set_cutoff(['C', 'C', 'C'], 0.2)
    except(ValueError):
        pass
    else:
        raise Exception("Three atoms allowed.")

    try:
        net.set_cutoff(['C', 'C'], 0)
    except(ValueError):
        pass
    else:
        raise Exception("Zero cutoff allowed.")

    try:
        net.set_cutoff(['C', 'C'], -0.2)
    except(ValueError):
        pass
    else:
        raise Exception("Negative cutoff allowed.")

    net.set_cutoff(['Li', 'N'], 0.3)

    assert net._cutoff[frozenset(['Li', 'N'])] == 0.3

    return


def test_decode():
    net = Network()
    net.replica.append({'traj': None, 'cmat': None, 'path': None,
                        'processed': False, 'network': None})
    net.n_atoms = 3
    net.frames.append([])

    net.replica[0]['cmat'] = np.zeros((200, 3, 3), dtype=int)
    net.replica[0]['cmat'][:, 0, 1] = 1
    idx = np.arange(0, 200, 10)
    net.replica[0]['cmat'][idx, 0, 2] = 1

    net.decode()

    assert np.all(net.replica[0]['cmat'][:, 0, 1] == 1)
    assert np.all(net.replica[0]['cmat'][:, 1, 2] == 0)
    assert np.all(net.replica[0]['cmat'][:, 0, 2] == 0)

    return