import os

import mdtraj as md
import numpy as np

from ..core import Network

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
    
    net = Network()
    net.add_replica(traj_path, top_path)
    net.generate_contact_matrix(ignore='Cl')

    assert (net.replica[0]['cmat'][:, [4, 5], :] == 0).all() and\
           (net.replica[0]['cmat'][:, :, [4, 5]] == 0).all()

    return


def test_generate_pairs():
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


def test_chemical_equations():
    net = Network()
    smiles_list = [('A.B.C', 0), ('A.B.D', 10), ('A.E.D', 50)]

    reactions = net.chemical_equations(-1, smiles_list)
    assert len(reactions) == 2, "List of equations not correct."

    assert reactions[0] == 'C --> D', "Incorrect first equation."
    assert reactions[1] == 'B --> E', "Incorrect second equation."

    return


def test_traj_to_topology():
    net1 = Network()
    net1.add_replica(traj_path, topology=top_path)
    table1, _ = net1.topology.top.to_dataframe()
    atoms1 = table1['element'].tolist()

    net2 = Network()
    net2.add_replica(traj_path)
    table2, _ = net2.topology.top.to_dataframe()
    atoms2 = table2['element'].tolist()

    assert atoms1 == atoms2, "Atom lists are not equivalent."

    new_top_name = os.path.join(testdir, 'test_case_topology.pdb')
    if os.path.exists(new_top_name):
        os.remove(new_top_name)
    return


def test_generate_SMILES():
    pass
    return


def test_draw_overall_network():
    pass
    return


def test_compute_distances():
    pass
    return


def test_reshape_to_square():
    pass
    return


def test_build_connections():
    pass
    return


def test_get_atoms():
    pass
    return


def test_build_cutoff():
    pass
    return


def test_bond_distance():
    pass
    return


def test_find_transition_frames():
    pass
    return


def test_build_network():
    pass
    return


def test_compile_networks():
    pass
    return


def test_build_all_networks():
    pass
    return
