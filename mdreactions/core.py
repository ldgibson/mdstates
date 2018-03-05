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


# Start of static functions
def clean(traj_file, top_file, states=[0, 1], start_p=[0.5, 0.5],
          trans_p=[[0.999, 0.001], [0.001, 0.999]],
          emission_p=[[0.6, 0.4], [0.4, 0.6]],
          periodic=True, CUTOFF=0.180):
    """
    Uses Hidden Markov Models and the Viterbi algorithm to clean the
    trajectory and determine the frame numbers for which reactions
    occurred.

    Default Values:
    - states: [0, 1]
    - start probability: [0.5, 0.5]
    - transition probability: [[0.999, 0.001],
                               [0.001, 0.999]]
    - emission probability: [[0.6, 0.4],
                             [0.4, 0.6]]
    - periodic: True
    - cutoff: 0.180 nm
    """

    if type(states) is not np.ndarray:
        states = np.array(states)
    if type(start_p) is not np.ndarray:
        start_p = np.array(start_p)
    if type(trans_p) is not np.ndarray:
        trans_p = np.array(trans_p)
    if type(emission_p) is not np.ndarray:
        emission_p = np.array(emission_p)

    traj = load_traj(traj_file, top_file)
    pairs = generate_pairs(traj.n_atoms)
    distances = compute_distances(traj, pairs, periodic=periodic)
    distances = change_shape(distances)
    cmat = build_connections(distances, CUTOFF=CUTOFF)
    ignore_list = generate_ignore_list(cmat)
    clean_cmat = clean_trajectory(cmat, ignore_list, states,
                                  start_p, trans_p, emission_p)
    rxn_frames = find_reaction_frames(clean_cmat)

    return rxn_frames


def load_traj(trajectory_file, topology_file):
    """
    Loads trajectory into python. Returns mdtraj.Trajectory object.
    """

    assert os.path.exists(trajectory_file), "Trajectory file does not exist."
    assert os.path.exists(topology_file), "Topology file does not exist."

    return md.load(trajectory_file, top=topology_file)


def generate_pairs(NUM_ATOMS):
    """
    Generates and returns the list of atom pairs for which interatomic
    distances will be calculated.
    """

    NUM_PAIRS = NUM_ATOMS * (NUM_ATOMS - 1) / 2
    pairs = []

    idx = 0
    for i in range(NUM_ATOMS-1):
        for j in range(i+1, NUM_ATOMS):
            pairs.append([i, j])
            idx = idx + 1
    return pairs


def compute_distances(traj, pairs, periodic):
    """Calculates and returns array of interatomic distances for all
    pairs passed"""
    return md.compute_distances(traj, pairs, periodic=periodic)


def change_shape(linear_mat):
    """
    Converts the shape of a 2D array into a 3D array: a square matrix
    for every frame.
    """

    NUM_ATOMS = int((1. + np.sqrt(1. + 8. * linear_mat.shape[1])) / 2.)
    NUM_FRAMES = linear_mat.shape[0]
    square_mat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS))
    upper_tri_idx = np.triu_indices(NUM_ATOMS, 1)

    for f in range(NUM_FRAMES):
        square_mat[f, upper_tri_idx[0], upper_tri_idx[1]] = linear_mat[f, :]

    return square_mat


def build_connections(distances, CUTOFF):
    """
    Classifies every atom pair as bonded or unbonded, depending on
    their interatomic distance and the cutoff value
    """

    assert CUTOFF > 0, "Negative cutoff values not allowed"

    assert distances.shape[1] == distances.shape[2],\
        "Matrix must be square for each frame"

    NUM_FRAMES = distances.shape[0]
    NUM_ATOMS = distances.shape[1]

    cmat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS), dtype=int)
    for i in range(NUM_ATOMS-1):
        for j in range(i+1, NUM_ATOMS):
            cmat[:, i, j] = np.where(distances[:, i, j] < CUTOFF, 1, 0)

    return cmat


def build_contact_matrix(NUM_FRAMES, NUM_ATOMS, rxn_matrix):
    """
    Depricated fuction to be removed.
    """
    cmat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS), dtype=int)
    for f in range(NUM_FRAMES):
        atom_idx = 0
        old_stop = 0
        for i in range(NUM_ATOMS-1, 0, -1):
            start = old_stop
            stop = start + i
            old_stop = stop
            cmat[f, atom_idx, NUM_ATOMS-i:] = rxn_matrix[f, start:stop]
            atom_idx = atom_idx + 1
    return cmat


def generate_ignore_list(cmat):
    """
    Generates a list of points in the contact matrix that should be
    ignored when using the Viterbi algorithm.  At each index in the
    ignore list, the value (or connectivity) does not change for all
    frames.
    """
    ignore_list = [[], []]
    NUM_ATOMS = cmat.shape[1]

    for i in range(NUM_ATOMS-1):
        for j in range(i+1, NUM_ATOMS):
            if (cmat[:, i, j] == 0).all():
                ignore_list[0].append([i, j])
            elif (cmat[:, i, j] == 1).all():
                ignore_list[1].append([i, j])
            else:
                pass
    return ignore_list


def viterbi(obs, states, start_p, trans_p, emit_p):
    """Viterbi algorithm used to decode a noisy trajectory"""

    V = [{}]
    for st in states:
        V[0][st] = {"log_prob": np.log10(start_p[st] * emit_p[st, obs[0]]),
                    "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["log_prob"] +
                              np.log10(trans_p[prev_st, st])
                              for prev_st in states)

            for prev_st in states:
                if V[t-1][prev_st]["log_prob"] + \
                        np.log10(trans_p[prev_st, st]) == max_tr_prob:
                    max_prob = max_tr_prob + np.log10(emit_p[st, obs[t]])
                    V[t][st] = {"log_prob": max_prob, "prev": prev_st}
                    break

    optimal_path = []

    # The highest probability
    max_final_prob = max(value["log_prob"] for value in V[-1].values())
    previous = None

    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["log_prob"] == max_final_prob:
            optimal_path.append(st)
            previous = st
            break

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        optimal_path.insert(0, V[t+1][previous]["prev"])
        previous = V[t+1][previous]["prev"]

    return optimal_path


def clean_trajectory(cmat, ignore_list, states, start_p, trans_p, emission_p):
    NUM_FRAMES = cmat.shape[0]
    NUM_ATOMS = cmat.shape[1]

    clean_cmat = np.zeros((NUM_FRAMES, NUM_ATOMS, NUM_ATOMS))
    for i in range(NUM_ATOMS-1):
        for j in range(i+1, NUM_ATOMS):
            if [i, j] in ignore_list[0]:
                clean_cmat[:, i, j] = 0
            elif [i, j] in ignore_list[1]:
                clean_cmat[:, i, j] = 1
            else:
                obs = cmat[:, i, j]
                clean_cmat[:, i, j] = viterbi(obs, states, start_p,
                                              trans_p, emission_p)
    return clean_cmat


def find_reaction_frames(cmat):
    """
    Loops over all frames of the contact matrix and checks if any
    changes occur.  Any frames in which a change occurred in the
    contact matrix are recorded.
    """
    NUM_FRAMES = cmat.shape[0]
    rxn_frames = []

    for f in range(1, NUM_FRAMES):
        if (cmat[f, :, :] == cmat[f-1, :, :]).all() is True:
            pass
        else:
            rxn_frames.append(f)

    return rxn_frames


def generate_SMILES(cmat, rxn_frames):
    """
    Generates a list of SMILES strings for each of the structures
    visited by the trajectory.
    """

    return SMILES
