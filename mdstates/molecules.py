import networkx as nx
import numpy as np
from rdkit import Chem


def contact_matrix_to_SMILES(cmat, atom_list):
    """Converts a contact matrix to a SMILES string.

    Parameters
    ----------
    cmat : numpy.ndarray
        Contact matrix describing connectivity in a molecule.
    atom_list : array-like
        Atom list with indices matching indices in contact matrix for
        atom type identification.

    Returns
    -------
    smiles : list of str
        List of SMILES string associated with every contact matrix.
    """

    mol = build_molecule(cmat, atom_list)

    mol = Chem.RemoveHs(mol)

    smiles = Chem.MolToSmiles(mol)

    return smiles


def build_molecule(cmat, atom_list):
    mol = Chem.RWMol()
    set_structure(mol, cmat, atom_list)
    radicals = build_radical_graph(mol)
    set_positive_charges(radicals)
    Chem.SanitizeMol(mol)
    estimate_bonds(radicals)
    Chem.SanitizeMol(mol)
    return mol


def set_structure(mol, cmat, atom_list):
    """Creates single bonds between atoms based on contact matrix.

    Creates single bonds between all connected atoms in contact
    matrix.

    Parameters
    ----------
    mol : rdkit.Chem.RWMol
    cmat : numpy.ndarray
    atom_list : list"""

    for atom in atom_list:
        mol.AddAtom(Chem.Atom(atom))

    for i, j in zip(*np.where(cmat[:, :] == 1)):
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)

    for at in mol.GetAtoms():
        at.SetNoImplicit(True)

    return


def estimate_bonds(graph):
    """Estimates bond order based on default atom valencies."""

    # Build connections of radicals
    for atom1, atom2 in graph.edges():
        atom1_radicals = atom1.GetNumRadicalElectrons()
        atom2_radicals = atom2.GetNumRadicalElectrons()

        if atom1_radicals >= 2 and atom2_radicals >= 2:
            graph.edges[atom1, atom2]['bond'].SetBondType(Chem.BondType.TRIPLE)
            atom1.SetNumRadicalElectrons(atom1_radicals - 2)
            atom2.SetNumRadicalElectrons(atom2_radicals - 2)
        elif atom1_radicals >= 1 and atom2_radicals >= 1:
            graph.edges[atom1, atom2]['bond'].SetBondType(Chem.BondType.DOUBLE)
            atom1.SetNumRadicalElectrons(atom1_radicals - 1)
            atom2.SetNumRadicalElectrons(atom2_radicals - 1)
        else:
            pass
    return


def build_radical_graph(mol):
    diff_valence = []
    pt = Chem.GetPeriodicTable()

    for atom in mol.GetAtoms():
        if atom.GetTotalDegree() < pt.GetDefaultValence(atom.GetSymbol()) or\
                atom.GetTotalDegree() > pt.GetDefaultValence(atom.GetSymbol()):
            diff_valence.append(atom)
        else:
            pass

    radicals = nx.Graph()

    for i, atom in enumerate(diff_valence):
        val = atom.GetTotalDegree() - pt.GetDefaultValence(atom.GetSymbol())
        if not radicals.nodes:
            radicals.add_node(atom, valence=val)
        else:
            radicals.add_node(atom, valence=val)

            bonds = []
            for node_atom in radicals:
                _bond = mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                node_atom.GetIdx())
                if _bond:
                    bonds.append([atom, node_atom, _bond])
                else:
                    pass

            for at1, at2, _bond in bonds:
                radicals.add_edge(at1, at2, bond=_bond)

    return radicals


def set_positive_charges(graph):

    pt = Chem.GetPeriodicTable()
    for atom in graph:
        default_valence = pt.GetDefaultValence(atom.GetSymbol())
        if atom.GetTotalDegree() > default_valence:
            charge = atom.GetTotalDegree() - default_valence
            atom.SetFormalCharge(charge)
        else:
            pass
    return


class Molecule:
    def __init__(self):
        self._atoms = []
        self._bonds = []
        return

    def add_atom(self, atom_sym, charge=0, local_multiplicity=1):
        self._atoms.append({'symbol': atom_sym, 'charge': charge,
                            'local_multiplicity': local_multiplicity})
        return

    @property
    def atoms(self):
        return [atom['symbol'] for atom in self._atoms]
