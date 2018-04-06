import re
import warnings

import mdtraj as md
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import pybel
from rdkit import Chem

from .data import bonds
from .graphs import combine_graphs, _prepare_graph
from .hmm import generate_ignore_list, viterbi
from .molecules import contact_matrix_to_SMILES
from .smiles import reduceSMILES, save_unique_SMILES, _break_ionic_bonds,\
    _radical_to_sp2

__all__ = ['Network']


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

        self._pairs = []
        self._cutoff = {}
        return

    def addreplica(self, trajectory, topology, **kwargs):
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
            assert self.replica[-1]['traj'].topology ==\
                md.load(topology).topology,\
                "All topologies must be the same."

            for rep in self.replica:
                assert rep['path'] != trajectory,\
                    "A trajectory from that location is already loaded."

        self.replica.append({'traj': None, 'cmat': None, 'path': None,
                             'processed': False, 'network': None})

        self.replica[-1]['traj'] = md.load(trajectory, top=topology,
                                           **kwargs)
        self.replica[-1]['path'] = trajectory

        # Add a sub-list for frames.
        self.frames.append([])

        # Set the topology.
        if self.topology is None:
            self.topology = topology
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

    def removereplica(self, rep_id):
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

    def generate_contact_matrix(self):
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
        """

        # Check if atom pairs have been determined.
        if not self._pairs:
            self._generate_pairs()
        else:
            pass

        # Check if cutoff matrix has been built.
        if len(self._cutoff) < len(self._pairs):
            self._build_cutoff()
        else:
            pass

        for i, rep in enumerate(self.replica):
            if rep['cmat'] is None:
                distances = self._compute_distances(i)
                distances = self._reshape_to_square(distances)
                rep['cmat'] = self._build_connections(distances)
            else:
                pass

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
        self._cutoff[frozenset(atoms)] = cutoff
        return

    def decode(self, n=10, states=[0, 1], start_p=[0.5, 0.5],
               trans_p=[[0.999, 0.001], [0.001, 0.999]],
               emission_p=[[0.60, 0.40], [0.40, 0.60]]):
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

        counter = 0

        for rep in self.replica:
            if rep['processed']:
                pass
            else:
                ignore_list = generate_ignore_list(rep['cmat'], n)

                for i in range(self.n_atoms - 1):
                    for j in range(i + 1, self.n_atoms):
                        if [i, j] in ignore_list[0]:
                            rep['cmat'][:, i, j] = 0
                        elif [i, j] in ignore_list[1]:
                            rep['cmat'][:, i, j] = 1
                        else:
                            counter += 1
                            rep['cmat'][:, i, j] =\
                                viterbi(rep['cmat'][:, i, j],
                                        states, start_p,
                                        trans_p, emission_p)

                # Mark that this replica's contact matrix
                # has been processed.
                rep['processed'] = True

        # After processing, locate all frames at which a
        # transition occurred and store it into `self.frames`
        self._find_transition_frames()

        print("{} iterations of Viterbi algorithm.".format(counter))
        return

    def _generate_pairs(self):
        """
        Generates the atom pairs for interatomic distance calculations.
        """
        for i in range(self.n_atoms - 1):
            for j in range(i + 1, self.n_atoms):
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
        >>> bar[0, :, :]  # first frame
        array([[0, 1, 2],
               [0, 0, 3],
               [0, 0, 0]])
        >>> bar[1, :, :]  # second frame
        array([[0, 4, 5],
               [0, 0, 6],
               [0, 0, 0]])
        """
        frames = linear_matrix.shape[0]
        square_matrix = np.zeros((frames, self.n_atoms, self.n_atoms))
        upper_tri_id = np.triu_indices(self.n_atoms, 1)

        for f in range(frames):
            square_matrix[f, upper_tri_id[0], upper_tri_id[1]] =\
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

        for i in range(self.n_atoms - 1):
            for j in range(i + 1, self.n_atoms):
                atom1 = self.atoms[i]
                atom2 = self.atoms[j]
                cmat[:, i, j] =\
                    np.where(distances[:, i, j] <
                             self._cutoff[frozenset([atom1, atom2])], 1, 0)
        return cmat

    def _get_atoms(self):
        """Generates the list of atoms in the trajectory."""

        table, bonds = self.replica[0]['traj'].top.to_dataframe()
        self.atoms = table['element'].tolist()
        return

    def _build_cutoff(self):
        """
        Builds the cutoff dictionary for all unique atom pairs.

        Finds the unique atoms in the system, then loops over
        every unique pair of atoms and checks the bond distance
        database for a value.

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
                        self._bond_distance(atom1, atom2)
        return

    def _bond_distance(self, atom1, atom2):
        """
        Queries a database to fetch a bond distance between two atoms.

        Parameters
        ----------
        atom1 : str
        atom2 : str

        Returns
        -------
        float
            Equilibrium bond distance for the 2 atoms that were passed.

        Raises
        ------
        LookupError
            If the atom pair is not present in the database.
        """
        pair = []
        pair.append(str(atom1) + '-' + str(atom2))
        pair.append(str(atom2) + '-' + str(atom1))
        loc_bool = [x in bonds.index for x in pair]
        if any(loc_bool):
            idx = loc_bool.index(True)
            return float(bonds.loc[pair[idx], 'distance']) * 0.13
        else:
            raise LookupError(pair[0] + " was not found in cutoff " +
                              "database. Please manually add it to " +
                              "the cutoff dictionary using " +
                              "Network.set_cutoff() and then rerun " +
                              "Network.generate_contact_matrix().")

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
            trans_frames = np.where(np.diff(rep['cmat'],
                                    axis=0).reshape((-1, self.n_atoms ** 2))
                                    .any(axis=1))[0]
            self.frames[rep_id] = list(trans_frames)
#            for f in range(1, rep['cmat'].shape[0]):
#                if (rep['cmat'][f, :, :] == rep['cmat'][f - 1, :, :]).all():
#                    pass
#                else:
#                    self.frames[rep_id].append(f)
        return

    def generate_SMILES_old(self, rep_id, tol=2, split_ions=True):
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

        warnings.warn("The only acceptable file format is XYZ from CP2K.",
                      RuntimeWarning)

        frames = self.frames[rep_id].copy()
        frames.insert(0, 0)

        smiles = []

        for mol in pybel.readfile("xyz", self.replica[rep_id]['path']):
            n_frame = int(re.findall("\d+", mol.title.split(',')[0])[0]) // 100
            if np.isclose(n_frame, frames, atol=2).any():
                # Read in frame and convert to SMILES.
                pybelsmiles = mol.write("smiles").split('\t')[0]

                # Break any Li-O ionic bonds.
                pybelsmiles = _break_ionic_bonds(pybelsmiles)

                # Change how SMILES is written using RDKit.
                mol = Chem.MolFromSmiles(pybelsmiles)
                new_smiles = Chem.MolToSmiles(mol)

                # Change any ternary radial carbons into sp2
                # hybridization to stay consistent.
                new_smiles = _radical_to_sp2(new_smiles)

                # Change SMILES back into RDKit format.
                mol = Chem.MolFromSmiles(new_smiles)
                smiles.append(Chem.MolToSmiles(mol))

            else:
                pass

        smiles = reduceSMILES(smiles)
        return smiles

    def generate_SMILES(self, rep_id, tol=2,
                        first_smiles='O=C1OCCO1.O=C1OCCO1.[Li]'):
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

        frames = self.frames[rep_id].copy()

        smiles = []
        smiles.append(first_smiles)

        for rep in self.replica:
            for f in range(rep['cmat'].shape[0]):
                if np.isclose(f, frames, atol=tol).any():
                    smi = contact_matrix_to_SMILES(rep['cmat'][f, :, :],
                                                   self.atoms)
                    smiles.append(smi)
                else:
                    pass

        smiles = reduceSMILES(smiles)
        return smiles

    def _build_network(self, smiles_list):
        """Builds the network from a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
        image_loc : str, optional
            Location of the folder containing all 2D SMILES structures.

        Returns
        -------
        network : networkx.DiGraph
        """

        save_unique_SMILES(smiles_list)

        network = nx.DiGraph()

        for i, smi in enumerate(smiles_list):
            # If the current SMILES string is missing in the network
            # graph, then add it.
            if not network.has_node(smi):
                # If the graph is empty, then add first node.
                if not network.nodes:
                    network.add_node(smi, rank=0, count=1, traj_count=1)
                else:
                    network.add_node(smi, count=1, traj_count=1)
                    network.add_edge(smiles_list[i - 1], smi, count=1,
                                     traj_count=1)

            # If the current SMILES string is present in the network.
            else:
                if network.has_edge(smiles_list[i - 1], smi):
                    network.edges[smiles_list[i - 1], smi]['count'] += 1
                else:
                    network.add_edge(smiles_list[i - 1], smi, count=1,
                                     traj_count=1)
        return network

    def drawfinalnetwork(self, layout='dot', **kwargs):
        """Builds networks for all replicas and combines them."""

        self._build_all_networks()

        # Compile all networks into a single graph.
        overall_network = nx.DiGraph()
        for rep in self.replica:
            overall_network = combine_graphs(overall_network, rep['network'])

        final = _prepare_graph(overall_network, **kwargs)
        self._draw_network(final, 'overall.png', layout=layout)
        return

    def _build_all_networks(self):
        """Builds networks for all replicas."""
        for rep_id, rep in enumerate(self.replica):
            smiles_list = self.generate_SMILES(rep_id)
            save_unique_SMILES(smiles_list)
            rep['network'] = self._build_network(smiles_list)
        return

    def _draw_network(self, nxgraph, filename, layout="dot"):
        pygraph = to_agraph(nxgraph)
        # pygraph.graph_attr['concentrate'] = 'true'
        pygraph.layout(layout)
        pygraph.draw(filename)
        return
