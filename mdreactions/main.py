import mdtraj as md
import numpy as np
import pandas as pd

from .data import bonds


def Network:

    def __init__(self):
        self.replica = {}
        self.atoms = None
        self.n_atoms = None
        self.pbc = True
        self.frames = ()

        self._pairs = []
        self._cutoff = {}
        return

    def addtraj(self, trajectory, topology, **kwargs):
        """
        Loads a trajectory into the class.

        Args:
            trajectory (str): File name of the trajectory file.
            topology (str): File name of the topology file.
            traj_id (int): ID of the loaded trajectory.
        """

        if not self.replica:
            rep_id = 0
            self.replica[rep_id] = {'traj': None, 'cmat': None}
        else:
            rep_id = list(self.replica.keys()).max() + 1
            self.replica[rep_id] = {'traj': None, 'cmat': None}
            
        self.replica[rep_id]['traj'] = md.load(trajectory, top=topology,
                                               **kwargs)

        # Set the number of atoms
        if self.n_atoms is None:
            self.n_atoms = self.replica[rep_id]['traj'].n_atoms
        else:
            pass

        if self.atoms is None:
            self._get_atoms(rep_id)
        else:
            pass

        return


    def generate_contact_matrix(self):

        if not self._pairs:
            self._generate_pairs()
        else:
            pass

        if not self._cutoff:
            self._build_cutoff()
        else:
            pass

        for rep_id in self.replica:
            if self.replica[rep_id]['cmat'] is None:
                distances = self._compute_distances(rep_id)
                distances = self._reshape_to_square(distances)
                self.replcia[rep_id]['cmat'] =\
                    self._build_connections(distances))
            else:
                pass

        return

    def _generate_pairs(self):
        """
        Generates the atom pairs for interatomic distance calculations.
        """
        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                self._pairs.append([i, j])

        return

    def _compute_distances(self, rep_id):
        """
        Computes the interatomic distances for all pairs of atoms.

        Args:
            rep_id (int): ID number of the replica of interest.

        Returns:
            numpy.ndarray: Interatomic distances for all specified atom
                pairs. Shape is (n_frames, n_pairs).
        """
        assert self._pairs, "Atom pairs not determined yet."

        return md.compute_distances(self.replica[rep_id]['traj'],
                                    self._pairs, periodic=self.pbc)

    def _reshape_to_square(self, linear_matrix):
        """
        Converts a row matrix into a square matrix.
        
        Takes each row and reshapes it to fit into the upper triangle
        of a square matrix, i.e. populating the upper off-diagonal
        elements.

        Args:
            linear_matrix (numpy.ndarray): Row of interatomic distances
                at every frame. Shape is (n_frames, n_pairs).

        Returns:
            numpy.ndarray: Reshaped matrix to be 3D. Each row is
                reshaped into a square matrix with only the upper off-
                diagonal elements populated for all frames.

        Example:
            >>> foo = np.array([[1, 2, 3],
                                [4, 5, 6]])

            >>> bar = self._reshape_to_square(foo)

            >>> bar[0, :, :]
            array([[0, 1, 2],
                   [0, 0, 3],
                   [0, 0, 0]])

            >>> bar[1, :, :]
            array([[0, 4, 5],
                   [0, 0, 6],
                   [0, 0, 0]])
        """
        frames = linear_matrix.shape[0]
        square_matrix = np.zeros((frames, self.n_atoms, self.n_atoms))
        upper_tri_id = np.triu_indices(self.n_atoms, 1)

        for f in range(frames): square_matrix[f, upper_tri_id[0], upper_tri_id[1]] =\
                linear_matrix[f, :]

        return square_matrix

    def _build_connections(self, distances):
        """
        Converts an array of distances to a contact matrix.

        Parameters
        ----------
        distances : numpy.ndarray
            Interatomic distances at all frames for all unique atom
            pairs.

        Returns
        -------
        cmat : numpy.ndarray
            Contact matrix at all frames
        """
        frames = distances.shape[0]

        cmat = np.zeros((frames, self.n_atoms, self.n_atoms), dtype=int)

        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                cmat[:, i, j] = np.where(distances[:, i, j] <
                                         self._cutoff[i, j], 1, 0)
        return cmat

    def _get_atoms(self, rep_id):
        table, bonds = self.replica[rep_id]['traj'].top.to_dataframe()
        self.atoms = table['element'].tolist()
        return

    def _build_cutoff(self):

        assert self.atoms, "Atom list is not yet compiled"

        atoms = self.atoms
        unique = []
        for atom in atoms:
            if atom not in unique:
                unique.append(atom)

        for i, atom1 in enumerate(unique):
            for atom2 in unique[i:]:
                if frozenset([atom1, atom2]) not in self._cutoff.keys():
                    self._cutoff[frozenset(atom1, atom2)] =\
                        self._bond_distance(atom1, atom2)
        return

    def _bond_distance(self, atom1, atom2):
        pair = []
        pair.append(str(atom1)+'-'+str(atom2))
        pair.append(str(atom2)+'-'+str(atom1))
        loc_bool = [x in bonds.index for x in pair]
        if all([y == False for y in loc_bool]):
            raise Exception('Bond distance not defined',
                            pair[0]+' not found in bond distance database. '+\
                            'Please add '+pair[0]+' cutoff distance to '+\
                            'database using Network.assign_cutoff().')
        else:
            idx = loc_bool.index(True)
            return float(bonds[pair[idx]])
