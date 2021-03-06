from numbers import Number
from itertools import combinations
# import re
from os.path import abspath, dirname, join

import mdtraj as md
from multiprocessing import Manager, Process
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import pandas as pd
import pybel

from .data import radii
from .graphs import combine_graphs, prepare_graph
from .hmm import generate_ignore_list, viterbi
from .hmm_cython import decode_cpp   # , viterbi_cpp
from .molecules import (contact_matrix_to_SMILES, cmat_to_structure,
                        molecule_to_json_string, json_string_to_molecule)
from .smiles import (remove_consecutive_repeats, save_unique_SMILES,
                     find_reaction)
from .util import find_nearest

__all__ = ['Network']


"""TODO:
    Build a new class to replace `Network.replica` attribute.
        This object should contain all of the information about
        the replica and should be easily indexed.
"""


class Network:
    """
    Analyze different states and transitions of an MD trajectory.

    Attributes
    ----------
    replica : dict
        Container for each trajectory and it's associated contact
        matrix. All trajectories must have matching topologies.
    atoms : list
        List of atoms for in topologoy.
    n_atoms : int
        Number of atoms in topology.
    pbc : bool
        Periodic boundary condition.
    frames : tuple
        Frames at which an transition is recorded.
    """

    def __init__(self):
        """Inits `Network` object."""
        self.replica = []
        self.atoms = None
        self.n_atoms = None
        self.pbc = True
        self.frames = []
        self.network = None
        self.topology = None
        self.first_smiles = None
        self.first_mol = None

        self._pairs = []
        self._cutoff = {}
        return

    def add_replica(self, trajectory, topology=None, **kwargs):
        """
        Adds a replica to the class object.

        Loads a trajectory into the `replica` attribute. Also adds an
        emtpy list to `frames` attribute. On the first replica added,
        attributes `n_atoms` and `atoms` are loaded from the topology
        information. All replicas must have the same topology and must
        have different `trajectory` paths.

        Parameters
        ----------
        trajectory : str
            File name of the trajectory file.
        topology : str
            File name of the topology file.
        **kwargs
            Additional keywords that can be sent to `mdtraj.load()`.

        Raises
        ------
        AssertionError
            If a new replica does not have a topology that matches the
            topology of the previously added replica.
        AssertionError
            If the trajectory path matches a previously loaded replica.
        """
        if len(self.replica) > 0:
            for rep in self.replica:
                assert rep['path'] != abspath(trajectory),\
                    "A trajectory from that location is already loaded."

        if isinstance(trajectory, list):
            for _ in range(len(trajectory)):
                self.replica.append({'traj': None, 'cmat': None, 'path': None,
                                     'processed': False, 'network': None,
                                     'structures': None})
        else:
            self.replica.append({'traj': None, 'cmat': None, 'path': None,
                                 'processed': False, 'network': None,
                                 'structures': None})
        if topology:
            pass
        else:
            topology = self._traj_to_topology(abspath(trajectory), 'xyz')

        if isinstance(trajectory, list):
            processes = []
            manager = Manager()
            rep_dict = []

            for i, traj in enumerate(trajectory):
                rep_dict.append(manager.dict(self.replica[i]))
                p = Process(target=self._add_replica_traj,
                            args=(rep_dict[i], traj, topology))
                processes.append(p)
                p.start()
                self.replica[i]['path'] = abspath(traj)
                # Add a sub-list for frames.
                self.frames.append([])

            for i, proc in enumerate(processes):
                proc.join()
                self.replica[i]['traj'] = rep_dict[i]['traj']

        else:
            self.replica[-1]['traj'] = md.load(trajectory, top=topology,
                                               **kwargs)
            self.replica[-1]['path'] = abspath(trajectory)
            # Add a sub-list for frames.
            self.frames.append([])

        # Set the topology.
        if self.topology is None:
            self.topology = md.load(topology)
        else:
            pass

        # Set the number of atoms.
        if self.n_atoms is None:
            self.n_atoms = self.replica[0]['traj'].n_atoms
        else:
            pass

        # Generate the atom list.
        if self.atoms is None:
            self._get_atoms()
        else:
            pass

        return

    def _add_replica_traj(self, rep, traj, top, **kwargs):
        rep['traj'] = md.load(traj, top=top, **kwargs)
        return

    def remove_replica(self, rep_id):
        """
        Removes a replica that was added previously.

        Parameters
        ----------
        rep_id : int

        Raises
        ------
        KeyError
            If `rep_id` does not match any replicas presently loaded.
        """

        # if rep_id not in self.replica.keys():
        if rep_id < len(self.replica) - 1:
            raise IndexError(str(rep_id) + ' is outside of the index range.')
        else:
            pass

        del self.replica[rep_id]
        return

    def generate_contact_matrix(self, cutoff_frac=1.4, ignore=None,
                                parallel=False):
        """
        Converts each trajectory frame to a contact matrix.

        The unique atom pairs are first generated if they have not been
        generated previously. Next, the cutoff dictionary for all the
        unique atom pairs is built. Lastly, for each replica that does
        not already have a contact matrix, it is generated by computing
        the interatomic distances for all pairs generated previously,
        reshapes the array into contact matrices for each frame, then
        converts the interatomic distances to a `1` or `0`, depending
        on the atom pair's cutoff.

        Parameters
        ----------
        cutoff_frac : float, optional
            Determines what each interatomic distance cutoff will be
            used for each atom pair. A `cutoff_frac` of 1.3 implies
            that for each atom pair, the cutoff distance will be 1.3
            times the equilibrium bond distance. Default is 1.3.
        ignore : str or list of str or int, optional
            If specified, all specified atoms will not be tracked in
            the contact matrix and will appear as unbonded, or 0, in
            all frames. Can be useful for species that are not
            covalently bonded, such as ions. Must pass either an atomic
            symbol, list of atomic symbols, or list of atom id's.
        """

        # Check if atom pairs have been determined.
        if not self._pairs:
            self._generate_pairs()
        else:
            pass

        # Check if cutoff matrix has been built.
        if len(self._cutoff) < len(self._pairs):
            self._build_cutoff(cutoff_frac)
        else:
            pass

        # Check if topology file has been converted to a SMILES string.
        if self.first_smiles:
            pass
        else:
            self._traj_to_smiles()

        # Build list of atoms to ignore.
        ignore_list = []
        if ignore:
            # Ex: ignore='Li'
            if isinstance(ignore, str):
                ignore_list = [idx for idx, atom in enumerate(self.atoms)
                               if atom == ignore]
            # Ex: ignore=[0, 2, 4] or ignore=['Li', 'H']
            elif isinstance(ignore, list):
                for atom in ignore:
                    if isinstance(atom, str):
                        sym_ignore_list = [idx for idx, at in
                                           enumerate(self.atoms)
                                           if at == atom]
                        for atom_id in sym_ignore_list:
                            ignore_list.append(atom_id)
                    elif isinstance(atom, Number):
                        ignore_list.append(atom)
                    else:
                        raise Exception("Must pass atom symbol or " +
                                        "index for it to be ignored.")
            else:
                raise Exception("'ignore' must be a string or a list.")
        else:
            pass

        if parallel:
            processes = []
            manager = Manager()
            rep_dict = []
            for i, rep in enumerate(self.replica):
                rep_dict.append(manager.dict(rep))
                p = Process(target=self._build_single_cmat,
                            args=(i, rep_dict[i]))
                processes.append(p)
                p.start()

            for i, proc in enumerate(processes):
                proc.join()
                self.replica[i]['cmat'] = rep_dict[i]['cmat']
                if ignore:
                    self.replica[i]['cmat'][ignore_list, :, :] = 0
                    self.replica[i]['cmat'][:, ignore_list, :] = 0
        else:
            for i, rep in enumerate(self.replica):
                if rep['cmat'] is None:
                    distances = self._compute_distances(i)
                    distances = self._reshape_to_square(distances)
                    rep['cmat'] = self._build_connections(distances)
                    if ignore:
                        rep['cmat'][ignore_list, :, :] = 0
                        rep['cmat'][:, ignore_list, :] = 0
                    else:
                        pass
                else:
                    pass
        return

    def _build_single_cmat(self, rep_id, rep):
        distances = self._compute_distances(rep_id)
        distances = self._reshape_to_square(distances)
        rep['cmat'] = self._build_connections(distances)
        return

    def set_cutoff(self, atoms, cutoff):
        """Assigns the cutoff for a pair of atoms.

        Parameters
        ----------
        atoms : list of str
            List containing two elemental symbols as strings.
        cutoff : float
            Cutoff value for the two elements in `atoms`.
        """
        if not isinstance(atoms, list):
            raise TypeError("Atoms must be passed in list.")
        else:
            pass

        if len(atoms) != 2:
            raise ValueError("Must pass two atoms.")
        else:
            pass

        if cutoff <= 0:
            raise ValueError("Cutoff must be positive and non-zero.")
        else:
            pass

        self._cutoff[frozenset(atoms)] = cutoff
        return

    def decode(self, n=10, states=[0, 1], start_p=[0.5, 0.5],
               trans_p=[[0.999, 0.001], [0.001, 0.999]],
               emission_p=[[0.60, 0.40], [0.40, 0.60]], min_lifetime=20,
               cores=1, use_python=False):
        """Uses Viterbi algorithm to clean the signal for each bond.

        Prior to processing each individual index in the contact
        matrix, an ignore list is constructed to reduce the number of
        times the Viterbi algorithm needs to be executed. If at any
        given index in the contact matrix there are more than `n`
        occurrences of the least common value at that index, then the
        signal for that index will be processed with the Viterbi
        algorithm.

        Parameters
        ----------
        n : int
            Threshold for determining if a bond should be decoded. If a
            bond has more than `n` occurrences of the least common
            state, then it will be processed with the Viterbi algorithm.
        states : list of int, optional
            Numerical representations of the states in the system.
            Default is [0, 1].
        start_p : list of float, optional
            Probabilities of starting in a particular hidden state.
            Default is [0.5, 0.5].
        trans_p : list of list of float, optional
            Probabilities of transitioning from one hidden state to
            another. Default is [[0.999, 0.001], [0.001, 0.999]].
        emission_p : list of of list of float, optional
            Probabilities of emitting an observable given the present
            hidden state. Default is [[0.6, 0.4], [0.4, 0.6]].
        """
        for rep in self.replica:
            assert rep['cmat'] is not None,\
                "Not all contact matrices have been generated."

        # Convert HMM parameters to ndarrays.
        if type(states) is not np.ndarray:
            states = np.array(states)
        if type(start_p) is not np.ndarray:
            start_p = np.array(start_p)
        if type(trans_p) is not np.ndarray:
            trans_p = np.array(trans_p)
        if type(emission_p) is not np.ndarray:
            emission_p = np.array(emission_p)

        for rep in self.replica:
            if rep['processed']:
                pass
            else:
                run_indices_i = []
                run_indices_j = []
                ignore_list = generate_ignore_list(rep['cmat'], n)

                # if cores == 1:
                for i in range(self.n_atoms - 1):
                    for j in range(i + 1, self.n_atoms):
                        if [i, j] in ignore_list[0]:
                            rep['cmat'][i, j, :] = 0
                        elif [i, j] in ignore_list[1]:
                            rep['cmat'][i, j, :] = 1
                        else:
                            if use_python:
                                rep['cmat'][i, j, :] =\
                                    viterbi(rep['cmat'][i, j, :], states,
                                            start_p, trans_p, emission_p)
                            else:
                                run_indices_i.append(i)
                                run_indices_j.append(j)

                if not use_python:
                    # Check if there is anything to decode.
                    if run_indices_i and run_indices_j:
                        rep['cmat'][run_indices_i, run_indices_j, :] = \
                            decode_cpp(rep['cmat'][run_indices_i,
                                                   run_indices_j, :],
                                       start_p, trans_p, emission_p, cores)
                    else:
                        pass
                else:
                    pass
                rep['processed'] = True

        # After processing, locate all frames at which a
        # transition occurred and store it into `self.frames`
        self._find_transition_frames()

        # Clean the transition frames such that only relavent reactive
        # events remain.
        self._clean_frames(min_lifetime=min_lifetime)

        return

    def generate_SMILES(self, rep_id, tol=10):
        """Generates list of SMILES strings from trajectory.

        Parameters
        ----------
        rep_id : int
            Replica identifier.
        tol : int, optional
            Trajectory frames that are +/- `tol` frames from transition
            frames will be converted to SMILES strings. Default is 2.

        Returns
        -------
        smiles : list of str
            List of SMILES strings compiled from all trajectory frames
            within the specified tolerance of transition frames.
        """

        frames = np.array(self.frames[rep_id].copy())
        cmat = self.replica[rep_id]['cmat']

        # if not frames:
        if frames.size == 0:
            num_frames = cmat.shape[2]
            last_smiles = contact_matrix_to_SMILES(cmat[:, :, -1], self.atoms)
            if last_smiles == self.first_smiles:
                return [(self.first_smiles, 0)]
            else:
                return [(self.first_smiles, 0),
                        (last_smiles, num_frames - 1)]
        else:
            pass

        smiles = []

        for f in range(cmat.shape[2]):
            if np.isclose(frames - f, 0, atol=tol).any():
                smi = contact_matrix_to_SMILES(cmat[:, :, f], self.atoms)
                frame = find_nearest(f, frames)
                smiles.append((smi, frame))
            else:
                pass

        last_smiles = contact_matrix_to_SMILES(cmat[:, :, -1], self.atoms)
        smiles.append((last_smiles, cmat.shape[2] - 1))

        reduced_smiles = remove_consecutive_repeats(smiles)

        if reduced_smiles[0][0] != self.first_smiles:
            reduced_smiles.insert(0, (self.first_smiles, 0))
        else:
            pass

        self.replica[rep_id]['smiles'] = reduced_smiles
        return reduced_smiles

    def get_structures(self, tol=10):
        for rep_id in range(len(self.replica)):
            if self.replica[rep_id]['structures'] is None:
                self.replica[rep_id]['structures'] =\
                    self.get_structures_from_replica(rep_id, tol)
            else:
                pass
        return

    def get_structures_from_replica(self, rep_id, tol):
        frames = np.array(self.frames[rep_id])
        cmat = self.replica[rep_id]['cmat']

        last_smiles, last_mol = cmat_to_structure(cmat[:, :, -1],
                                                  self.atoms)
        num_frames = cmat.shape[2]
        # Handles if there are no recorded transitions.
        if frames.size == 0:
            # If the structure has not changed.
            if last_smiles == self.first_smiles:
                return pd.DataFrame({'smiles': [self.first_smiles],
                                     'molecule': [self.first_mol],
                                     'frame': [0],
                                     'transition_frame': [0]})
            # If a transition happend so early that
            # it was never detected and recorded.
            else:
                return pd.DataFrame({'smiles': [self.first_smiles,
                                                last_smiles],
                                     'molecule': [self.first_mol,
                                                  last_mol],
                                     'frame': [0, num_frames - 1],
                                     'transition_frame': [0, 0]})
        else:
            pass

        structures = pd.DataFrame(columns=['smiles', 'molecule', 'frame',
                                           'transition_frame'])
        for f in range(cmat.shape[2]):
            # If the current frame is within `tol` of any of the
            # transition frames in `frames`.
            if np.isclose(frames - f, 0, atol=tol).any():
                smi, mol = cmat_to_structure(cmat[..., f], self.atoms)
                transition_frame = find_nearest(f, frames)
                new_row = pd.DataFrame({'smiles': smi,
                                        'molecule': mol,
                                        'frame': f,
                                        'transition_frame': transition_frame},
                                       index=[0])
                structures = structures.append(new_row, ignore_index=True)
            else:
                pass

        # Adds the last structure to the end.
        last_row = pd.DataFrame({'smiles': last_smiles,
                                 'molecule': last_mol,
                                 'frame': num_frames - 1,
                                 'transition_frame': num_frames - 1},
                                index=[0])
        structures = structures.append(last_row, ignore_index=True)

        reduced_structures = remove_consecutive_repeats(structures)

        if reduced_structures.loc[0, 'smiles'] != self.first_smiles:
            first_row = pd.DataFrame({'smiles': [self.first_smiles],
                                      'molecule': [self.first_mol],
                                      'frame': [0],
                                      'transition_frame': [0]})
            reduced_structures = pd.concat([first_row, reduced_structures],
                                           ignore_index=True)
        return reduced_structures

    def draw_overall_network(self, filename='overall.png', exclude=[],
                             SMILES_loc='SMILESimages', use_LR=False,
                             use_graphviz=False, tree_depth=None, **kwargs):
        """Builds networks for all replicas and combines them.

        Parameters
        ----------
        filename : str
            Name of final graph image.
        exclude : list of int, optional
            List of replica IDs to exclude from the overall graph
            generation.
        SMILES_loc : str
            Name of the directory in which SMILES images will be saved.
        """

        kwargs_to_pass = dict()
        if 'tol' in kwargs:
            kwargs_to_pass.update(tol=kwargs['tol'])
        else:
            pass

        self.build_all_networks(**kwargs_to_pass)

        for rep in self.replica:
            smiles_list = rep['structures']['smiles'].tolist()
            save_unique_SMILES(smiles_list)

        # print("Saving SMILES images to: {}".format(abspath(SMILES_loc)))
        compiled = self._compile_networks(exclude=exclude)
        self.network = compiled
        final = prepare_graph(compiled, root_node=self.first_smiles, **kwargs)

        # Remove all nodes that are beyond a threshold tree depth.
        if tree_depth:
            lengths = nx.shortest_path_length(final, source=self.first_smiles)
            for node in lengths:
                if lengths[node] > tree_depth:
                    final.remove_node(node)
                else:
                    pass
        else:
            pass

        print("Saving network to: {}".format(abspath(filename)))
        if use_graphviz:
            self._draw_network_with_graphviz(final, filename=filename)
        else:
            self._draw_network(final, filename=filename, use_LR=use_LR)
        return

    def chemical_equations(self, rep_id, *args):
        """
        Converts a list of SMILES strings to chemical equations.

        Parameters
        ----------
        rep_id : int
            Replica ID number. If equal to `-1`, then ignores the value and
            takes list from `*args`.

        Returns
        -------
        list of str
            List containing chemical equations that include only
            reacting species.
        """

        if rep_id == -1 and args:
            smiles_list = args[0]
        else:
            assert self.replica[rep_id]['structures'],\
                "Structures list is empty."
            smiles_list = self.replica[rep_id]['structures']['smiles'].tolist()

        chem_eq_list = []
        for i, smi in enumerate(smiles_list):
            if i == 0:
                continue
            chem_eq_list.append(find_reaction(smiles_list[i - 1], smi))

        return chem_eq_list

    def _generate_pairs(self):
        """
        Generates the atom pairs for interatomic distance calculations.
        """

        for i, j in combinations(range(self.n_atoms), 2):
            self._pairs.append([i, j])
        return

    def _compute_distances(self, rep_id):
        """
        Computes the interatomic distances for all pairs of atoms.

        Parameters
        ----------
        rep_id : int
            ID number of the replica of interest.

        Returns
        -------
        numpy.ndarray
            Interatomic distances for all specified atom pairs. Shape
            is (n_frames, n_pairs).

        Raises
        ------
        IndexError
            If `rep_id` is not an index in `replica`.
        AssertionError
            If the trajectory associated with `rep_id` does not exist.
        AssertionError
            If the atom pairs have not been determined.
        """
        if rep_id > len(self.replica) - 1:
            raise IndexError(str(rep_id) + " is outside of index range.")
        assert self.replica[rep_id]['traj'],\
            "Trajectory does not exist."
        assert self._pairs, "Atom pairs not determined yet."

        return md.compute_distances(self.replica[rep_id]['traj'],
                                    self._pairs, periodic=self.pbc)

    def _reshape_to_square(self, linear_matrix):
        """
        Converts a row matrix into a square matrix.

        Takes each row and reshapes it to fit into the upper triangle
        of a square matrix, i.e. populating the upper off-diagonal
        elements.

        Parameters
        ----------
        linear_matrix : numpy.ndarray
            Row of interatomic distances at every frame. Shape is
            (n_frames, n_pairs).

        Returns
        -------
        square_matrix : numpy.ndarray
            Reshaped matrix to be 3D. Each row is reshaped into a square
            matrix with only the upper off-diagonal elements populated
            for all frames.

        Example
        -------
        >>> foo = np.array([[1, 2, 3],
        ...                 [4, 5, 6]])
        >>> bar = self._reshape_to_square(foo)
        >>> bar[:, :, 0]  # first frame
        array([[0, 1, 2],
               [0, 0, 3],
               [0, 0, 0]])
        >>> bar[:, :, 1]  # second frame
        array([[0, 4, 5],
               [0, 0, 6],
               [0, 0, 0]])
        """
        frames = linear_matrix.shape[0]
        square_matrix = np.zeros((self.n_atoms, self.n_atoms, frames))
        upper_tri_id = np.triu_indices(self.n_atoms, 1)

        for f in range(frames):
            square_matrix[upper_tri_id[0], upper_tri_id[1], f] =\
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
        frames = distances.shape[2]

        cmat = np.zeros((self.n_atoms, self.n_atoms, frames), dtype=np.int32)

        for i in range(self.n_atoms - 1):
            for j in range(i + 1, self.n_atoms):
                atom1 = self.atoms[i]
                atom2 = self.atoms[j]
                cmat[i, j, :] =\
                    np.where(distances[i, j, :] <
                             self._cutoff[frozenset([atom1, atom2])], 1, 0)
        return cmat

    def _get_atoms(self):
        """Generates the list of atoms in the trajectory."""

        table, _ = self.replica[0]['traj'].top.to_dataframe()
        self.atoms = table['element'].tolist()
        return

    def _build_cutoff(self, cutoff_frac=1.3):
        """
        Builds the cutoff dictionary for all unique atom pairs.

        Finds the unique atoms in the system, then loops over
        every unique pair of atoms and checks the bond distance
        database for a value.

        Parameters
        ----------
        cutoff_frac : float, optional
            See `generate_contact_matrix` for parameter description.

        Raises
        ------
        AssertionError
            If the atom list is not compiled.
        """

        assert self.atoms, "Atom list is not yet compiled"

        atoms = self.atoms
        unique = []
        for atom in atoms:
            if atom not in unique:
                unique.append(atom)

        for i, atom1 in enumerate(unique):
            for atom2 in unique[i:]:
                if frozenset([atom1, atom2]) not in self._cutoff.keys():
                    self._cutoff[frozenset([atom1, atom2])] =\
                        self._bond_distance(atom1, atom2, frac=cutoff_frac)
        return

    def _bond_distance(self, atom1, atom2, frac=1.4):
        """
        Queries a database to fetch a bond distance between two atoms.

        Parameters
        ----------
        atom1, atom2 : str
        frac : float, optional
            See `generate_contact_matrix` for parameter description.

        Returns
        -------
        float
            Equilibrium bond distance for the 2 atoms that were passed.

        Raises
        ------
        LookupError
            If the atom pair is not present in the database.
        """
        try:
            r1 = radii.loc[atom1, 'single']
        except(Exception):
            raise Exception(atom1 + ' does not have a covalent' +
                            ' radius in database.')
        try:
            r2 = radii.loc[atom2, 'single']
        except(Exception):
            raise Exception(atom2 + ' does not have a covalent' +
                            ' radius in database.')
        bond_distance = ((r1 + r2) / 1000) * frac
        return bond_distance

    def _find_transition_frames(self):
        """Finds transitions in the processed contact matrix.

        Checks if two consecutive contact matrices are not equivalent.
        If they are not, then a transition likely occured and the frame
        number is recorded.

        Raises
        ------
        AssertionError
            If any contact matrices have not been constructed or
            processed yet.
        AssertionError
            If the number of empty lists in `self.frames` is less than
            the number of replicas.
        """

        for rep in self.replica:
            assert rep['cmat'] is not None,\
                "Not all contact matrices have been constructed."
            assert rep['processed'],\
                "Not all replica contact matrices have been processed."

        assert len(self.frames) == len(self.replica),\
            "Number of sets of frames does not equal number of replicas."

        for rep_id, rep in enumerate(self.replica):
            trans_frames = np.where(np.diff(rep['cmat'])
                                    .reshape((self.n_atoms ** 2, -1)))[1]
            self.frames[rep_id] = list(trans_frames)
        return

    def _build_network(self, rep_id):
        """Builds the network from a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list of tuple of str and int
            List of tuples containing the SMILES string and the nearest
            transition frame.

        Returns
        -------
        network : networkx.DiGraph
        """

        network = nx.DiGraph()

        structures = self.replica[rep_id]['structures']

        for i, row in structures.iterrows():
            smi = row['smiles']
            f = row['transition_frame']
            # If the current SMILES string is missing in the network
            # graph, then add it.
            if not network.has_node(smi):
                # If the graph is empty, then add first node.
                if not network.nodes:
                    network.add_node(smi, count=1, traj_count=1)
                else:
                    network.add_node(smi, count=1, traj_count=1)
                    prev_smiles = structures.loc[i - 1, 'smiles']
                    network.add_edge(prev_smiles, smi, count=1,
                                     traj_count=1, frames=[])
                    network.edges[prev_smiles, smi]['frames'].append(f)

            # If the current SMILES string is present in the network.
            else:
                network.node[smi]['count'] += 1
                prev_smiles = structures.loc[i - 1, 'smiles']
                if network.has_edge(structures.loc[i - 1, 'smiles'], smi):
                    network.edges[prev_smiles, smi]['count'] += 1
                    network.edges[prev_smiles, smi]['frames'] .append(f)
                else:
                    network.add_edge(prev_smiles, smi, count=1,
                                     traj_count=1, frames=[])
                    network.edges[prev_smiles, smi]['frames'].append(f)
        return network

    def _compile_networks(self, exclude=[]):
        """Compiles all replica networks into a single overall network.

        Parameters
        ----------
        exclude : list
            List of replica IDs to exclude from overall network
            compilation.

        Returns
        -------
        final : networkx.DiGraph
            Compiled network.
        """
        overall_network = nx.DiGraph()
        for i, rep in enumerate(self.replica):
            if i in exclude:
                pass
            else:
                overall_network = combine_graphs(overall_network,
                                                 rep['network'])

        return overall_network

    def build_all_networks(self, **kwargs):
        """Builds networks for all replicas."""
        self.get_structures()
        for rep_id, rep in enumerate(self.replica):
            # smiles_list = self.generate_SMILES(rep_id, **kwargs)
            rep['network'] = self._build_network(rep_id)
        return

    def _draw_network(self, nxgraph, filename, layout="dot", write=True,
                      use_LR=False):
        try:
            import pygraphviz
        except(ModuleNotFoundError):
            raise ModuleNotFoundError("pygraphviz not installed.")

        pygraph = to_agraph(nxgraph)

        if use_LR:
            pygraph.graph_attr['rankdir'] = 'LR'
        else:
            pass

        pygraph.add_subgraph([self.first_smiles], rank='source')
        pygraph.layout(layout)
        if write:
            pygraph.write("input.dot")
        else:
            pass
        pygraph.draw(filename)
        return

    def _draw_network_with_graphviz(self, overall, filename="overall",
                                    format='png'):
        """Draw with Python Graphviz instead of PyGraphviz."""
        try:
            from graphviz import Digraph
        except(ModuleNotFoundError):
            raise ModuleNotFoundError("graphviz not installed.")
        g = Digraph('G', filename='graph.gv', format=format)
        # self.first_smiles = 'O=C1OCCO1.O=C1OCCO1.[Li]'
        for n, data in overall.nodes(data=True):
            if n == self.first_smiles:
                with g.subgraph(name='top') as top:
                    top.graph_attr.update(rank='source')
                    top.node(n, image=data['image'])
            else:
                g.node(n, image=data['image'])

        for u, v, data in overall.edges(data=True):
            if 'dir' in data:
                g.edge(u, v, dir='both')
            elif 'style' in data:
                g.edge(u, v, style=data['style'])
            elif 'penwidth' in data:
                g.edge(u, v, penwidth=data['penwidth'])
            else:
                g.edge(u, v)

        g.node_attr.update(label="")

        g.render(filename=filename)

        return

    def _clean_frames(self, min_lifetime):

        for frames in self.frames:
            bad_frames = []
            for i, f in enumerate(frames):
                if f == frames[-1]:
                    break
                else:
                    if frames[i + 1] - f < min_lifetime:
                        bad_frames.append(f)
                    else:
                        pass

            for bad_frame in bad_frames:
                frames.remove(bad_frame)

        return

    def _traj_to_smiles(self):
        """Generates the SMILES for the starting point in trajectory."""
        distances = md.compute_distances(self.replica[0]['traj'][0],
                                         self._pairs, periodic=self.pbc)
        distances = self._reshape_to_square(distances)
        cmat = self._build_connections(distances)
        self.first_smiles, self.first_mol = cmat_to_structure(cmat[..., 0],
                                                              self.atoms)
        return

    def _traj_to_topology(self, traj, format='xyz'):
        """Converts the first frame in trajectory to the topology file."""
        mol = next(pybel.readfile(format=format, filename=traj))

        # Grab the name of the file without extension
        # from the absolute path.
        filename = traj.split('/')[-1].split('.')[:-1]
        # Add back any '.' to the filename if they existed.
        filename = '.'.join(filename)
        top_name = filename + '_topology.pdb'
        parent_dir = dirname(traj)
        topology_path = join(parent_dir, top_name)
        mol.write('pdb', topology_path, overwrite=True)
        return topology_path

    def save(self, name):
        """Saves the current state of the reaction network as a txt.

        This saves the SMILES list and frames that each SMILES state
        is associated with in a text file.

        Parameters
        ----------
        name : str
            Name of the checkpoint file.
        """
        for rep in self.replica:
            if rep['structures'] is None:
                self.get_structures(tol=10)
                break
            else:
                pass

        with open(name + '.txt', 'w') as f:
            f.write('{}\n'.format(self.first_smiles))
            for rep_id, rep in enumerate(self.replica):
                f.write('replica{}\n'.format(rep_id))
                for i, row in rep['structures'].iterrows():
                    mol_graph = molecule_to_json_string(row['molecule'])
                    smiles = row['smiles']
                    frame = row['frame']
                    transition_frame = row['transition_frame']
                    f.write("{}|{}|{}|{}\n".format(frame, transition_frame,
                                                   smiles, mol_graph))
        return

    def load(self, name):
        """Loads the network from a checkpoint file.

        Parameters
        ----------
        name : str
            Path to the checkpoint file.
        """
        column_names = ['frame', 'transition_frame', 'smiles', 'molecule']
        with open(name, 'r') as f:
            self.first_smiles = f.readline().strip('\n')
            for line in f.readlines():
                if line.startswith('replica'):
                    # rep_id = int(re.search('(?<=replica)\d*', line).group(0))
                    self.replica.append({'traj': None, 'cmat': None,
                                         'path': None, 'processed': False,
                                         'network': None, 'structures': None})
                    self.replica[-1]['structures'] =\
                        pd.DataFrame(columns=column_names)
                else:
                    data = line.strip('\n').split('|')
                    data_dict = dict((key, val) for key, val
                                     in zip(column_names, data))
                    df = pd.DataFrame(data_dict, index=[0])

                    # Cast `frame` and `transition_frame` as integers.
                    df['frame'] = pd.to_numeric(df['frame'],
                                                downcast='integer')
                    df['transition_frame'] =\
                        pd.to_numeric(df['transition_frame'],
                                      downcast='integer')
                    # Convert json string back to rdkit molecule.
                    df['molecule'] =\
                        [json_string_to_molecule(df.loc[0, 'molecule'])]

                    self.replica[-1]['structures'] =\
                        self.replica[-1]['structures']\
                        .append(df, ignore_index=True)
        return

    def build_from_load(self, filename='overall.png', exclude=[],
                        SMILES_loc='SMILESimages', use_LR=False,
                        use_graphviz=False, tree_depth=None, layout='dot',
                        **kwargs):
        for rep in self.replica:
            rep['network'] = self._build_network(rep['smiles'])

        compiled = self._compile_networks(exclude=exclude)
        self.network = compiled
        final = prepare_graph(compiled, root_node=self.first_smiles, **kwargs)

        # Remove all nodes that are beyond a threshold tree depth.
        if tree_depth:
            lengths = nx.shortest_path_length(final, source=self.first_smiles)
            for node in lengths:
                if lengths[node] > tree_depth:
                    final.remove_node(node)
                else:
                    pass
        else:
            pass

        print("Saving network to: {}".format(abspath(filename)))
        if use_graphviz:
            self._draw_network_with_graphviz(final, filename=filename)
        else:
            self._draw_network(final, filename=filename, use_LR=use_LR,
                               layout=layout)
        return
