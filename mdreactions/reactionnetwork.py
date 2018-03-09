import os

import mdtraj as md
import numpy as np


class ReactionNetwork:
    """Create a ReactionNetwork object"""

    # Maybe add a Trajectory subclass
    # Maybe add a ContactMatrix subclass

    def __init__(self, trajectory, topology, CUTOFF, periodic=True):
        self.traj = md.load(trajectory, top=topology)

        pairs = self.generate_pairs()
        distances = self.compute_distances(self, periodic=periodic)
        sq_distances = self.change_shape(distances)
        cmat = self.build_connections(sq_distances, CUTOFF)


    def generate_pairs(self):
        """
        Generates and returns the list of atom pairs for which interatomic
        distances will be calculated.
        """

        NUM_ATOMS = self.traj.n_atoms
        NUM_PAIRS = NUM_ATOMS * (NUM_ATOMS - 1) / 2
        pairs = []

        idx = 0
        for i in range(NUM_ATOMS-1):
            for j in range(i+1, NUM_ATOMS):
                pairs.append([i, j])
                idx = idx + 1
        return pairs


    def compute_distances(self, pairs, periodic):
        """
        Calculates and returns array of interatomic distances for all
        pairs passed.
        """
        return md.compute_distances(self.traj, pairs, periodic=periodic)


    def change_shape(self, linear_mat):
        """
        Converts the shape of a 2D array into a 3D array: a square matrix
        for every frame.
        """

        NUM_ATOMS = self.traj.n_atoms
        NUM_FRAMES = self.traj.n_frames
        square_mat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS))
        upper_tri_idx = np.triu_indices(NUM_ATOMS, 1)

        for f in range(NUM_FRAMES):
            square_mat[f, upper_tri_idx[0], upper_tri_idx[1]] = \
                linear_mat[f, :]

        return square_mat


    def build_connections(self, distances, CUTOFF):
        """
        Classifies every atom pair as bonded or unbonded, depending on
        their interatomic distance and the cutoff value
        """

        assert CUTOFF > 0, "Negative cutoff values not allowed"

        assert distances.shape[1] == distances.shape[2],\
            "Matrix must be square for each frame"

        # NUM_FRAMES = distances.shape[0]
        # NUM_ATOMS = distances.shape[1]
        NUM_FRAMES = self.traj.n_frames
        NUM_ATOMS = self.traj.n_atoms

        cmat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS), dtype=int)
        for i in range(NUM_ATOMS-1):
            for j in range(i+1, NUM_ATOMS):
                cmat[:, i, j] = np.where(distances[:, i, j] < CUTOFF, 1, 0)

        return cmat
