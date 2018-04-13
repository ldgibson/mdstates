import os.path

import mdtraj as md
# import numpy as np

from ..main import Network

currentdir = os.path.abspath(__file__)
testdir = os.path.join(currentdir, 'test_cases')
traj_path = os.path.join(testdir, 'test_case.xyz')
top_path = os.path.join(testdir, 'test_top.pdb')


def test_Network():
    net = Network()

    def test_addreplica():
        net.__init__()
        net.addreplica(traj_path, top_path)
        assert isinstance(net.replica[0]['traj'], md.Trajectory), \
            '"traj" is not of type: mdtraj.Trajectory'

        # Test assertion for trajectory file existence
        try:
            net.addreplica('bad_traj_file', top_path)
        except (Exception):
            pass
        else:
            raise Exception('Failed assertion',
                            'Bad trajectory file was accepted without error')

        # Test assertion for topology file existence
        try:
            net.addreplica(traj_path, 'bad_top_file')
        except (Exception):
            pass
        else:
            raise Exception('Failed assertion',
                            'Bad topology file was accepted without error')
        return
