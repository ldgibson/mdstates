import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from .reactionoperator import BEMatrix

from .util import json_to_string, load_json_from_string


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
    smiles : str
        SMILES string of molecule built from contact matrix.
    """

    # Build the molecule
    mol, molHs = build_molecule(cmat, atom_list)

    # Generate SMILES string from molecule
    smiles = Chem.MolToSmiles(mol)

    return (smiles, molHs)


def cmat_to_structure(cmat, atom_list):
    """Converts a contact matrix to a SMILES string and molecule.

    Parameters
    ----------
    cmat : numpy.ndarray
        Contact matrix describing connectivity in a molecule.
    atom_list : array-like
        Atom list with indices matching indices in contact matrix for
        atom type identification.

    Returns
    -------
    smiles : str
        SMILES string of molecule built from contact matrix.
    mol : rdkit.Chem.Mol
        Molecule built from contact matrix.
    """

    # Build the molecule
    mol = build_molecule(cmat, atom_list, with_hydrogens=True)

    # Generate SMILES string from molecule
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

    return smiles, mol


def build_molecule(cmat, atom_list, with_hydrogens=False):
    """Builds a molecule from a contact matrix and list of atoms.

    Parameters
    ----------
    cmat : numpy.ndarray
        Contact matrix defining the connectivity of the molecule(s).
    atom_list : list
        List of atoms with indices that correspond to those in the
        contact matrix.

    Returns
    -------
    mol : rdkit.Chem.rdchem.RWMol
    """

    mol = Chem.RWMol()

    # Connect atoms with single bonds.
    set_structure(mol, cmat, atom_list)

    # Set positive charges to atoms with excess valence.
    set_positive_charges(mol)
    Chem.SanitizeMol(mol)

    # Estimate any double and triple bonds.
    estimate_bonds(mol)
    Chem.SanitizeMol(mol)

    if with_hydrogens:
        return mol
    else:
        # Remove any unnecessary hydrogens.
        mol = Chem.RemoveHs(mol)
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

    for i, atom in enumerate(atom_list):
        newatom = Chem.Atom(atom)
        newatom.SetNoImplicit(True)
        newatom.SetIntProp('index', i)
        mol.AddAtom(newatom)

    for i, j in zip(*np.where(cmat[:, :] == 1)):
        if mol.GetAtomWithIdx(int(i)).GetSymbol() == 'Li':
            continue
        elif mol.GetAtomWithIdx(int(j)).GetSymbol() == 'Li':
            continue
        else:
            pass

        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)

    # for at in mol.GetAtoms():
        # at.SetNoImplicit(True)

    return


def estimate_bonds(mol):
    """Estimates bond order based on default atom valencies."""

    # Build connections of radicals
    for bond in mol.GetBonds():
        atom1_radicals = bond.GetBeginAtom().GetNumRadicalElectrons()
        atom2_radicals = bond.GetEndAtom().GetNumRadicalElectrons()

        if atom1_radicals >= 2 and atom2_radicals >= 2:
            bond.SetBondType(Chem.BondType.TRIPLE)
            bond.GetBeginAtom().SetNumRadicalElectrons(atom1_radicals - 2)
            bond.GetEndAtom().SetNumRadicalElectrons(atom2_radicals - 2)
        elif atom1_radicals >= 1 and atom2_radicals >= 1:
            bond.SetBondType(Chem.BondType.DOUBLE)
            bond.GetBeginAtom().SetNumRadicalElectrons(atom1_radicals - 1)
            bond.GetEndAtom().SetNumRadicalElectrons(atom2_radicals - 1)
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

    for atom in diff_valence:
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


def set_positive_charges(mol):

    pt = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        default_valence = pt.GetDefaultValence(atom.GetSymbol())
        if atom.GetTotalDegree() > default_valence:
            charge = atom.GetTotalDegree() - default_valence
            atom.SetFormalCharge(charge)
        else:
            pass
    return


def molecule_to_contact_matrix(mol):
    """Converts an RDKit molecule into a contact matrix.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol

    Returns
    -------
    mdstates.reactionoperator.BEMatrix
        Contact matrix for the given molecule where `1` denotes a
        single bond, `2` a double bond, and `3` a triple bond."""

    # mol = Chem.AddHs(mol)
    Chem.Kekulize(mol)
    dim = mol.GetNumAtoms(onlyExplicit=False)
    cmat = np.zeros((dim, dim), dtype=np.int32)
    atom_list = np.empty(dim, dtype='U8')
    for atom in mol.GetAtoms():
        try:
            idx = atom.GetIntProp('index')
        except(KeyError):
            idx = atom.GetIdx()
        atom_list[idx] = atom.GetSymbol()

    for bond in mol.GetBonds():
        try:
            atom1_idx = bond.GetBeginAtom().GetIntProp('index')
            atom2_idx = bond.GetEndAtom().GetIntProp('index')
        except(KeyError):
            atom1_idx = bond.GetBeginAtom().GetIdx()
            atom2_idx = bond.GetEndAtom().GetIdx()

        if atom1_idx < atom2_idx:
            i = atom1_idx
            j = atom2_idx
        else:
            j = atom1_idx
            i = atom2_idx

        if bond.GetBondType() == BondType.SINGLE:
            cmat[i, j] = 1
        elif bond.GetBondType() == BondType.DOUBLE:
            cmat[i, j] = 2
        elif bond.GetBondType() == BondType.TRIPLE:
            cmat[i, j] = 3
        else:
            raise Exception("Unrecognized BondType.")
    bemat = BEMatrix(cmat, atom_list.tolist())
    return bemat


def molecule_to_nxgraph(mol):
    """Converts an rdkit molecule to a networkx graph."""
    Chem.Kekulize(mol)
    gmol = nx.Graph()
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        idx = atom.GetIdx()
        gmol.add_node(idx, symbol=symbol)

    for bond in mol.GetBonds():
        at1_id = bond.GetBeginAtom().GetIdx()
        at2_id = bond.GetEndAtom().GetIdx()

        if bond.GetBondType() == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            bond_order = 3
        else:
            raise Exception("Bond type not recognized.")

        gmol.add_edge(at1_id, at2_id, bond_order=bond_order)
    return gmol


def nxgraph_to_molecule(graph):
    """Converts a networkx graph to an rdkit molecule."""
    mol = Chem.RWMol()
    for at, data in sorted(graph.nodes(data=True)):
        idx = mol.AddAtom(Chem.Atom(data['symbol']))
        if at != idx:
            raise Exception("Wrong atom index assigned.")
        else:
            pass

    for at1, at2, data in graph.edges(data=True):
        if data['bond_order'] == 1:
            bond_order = BondType.SINGLE
        elif data['bond_order'] == 2:
            bond_order = BondType.DOUBLE
        elif data['bond_order'] == 3:
            bond_order = BondType.TRIPLE
        mol.AddBond(at1, at2, order=bond_order)

    Chem.SanitizeMol(mol)
    return mol


def nxgraph_to_json(graph):
    """Converts a networkx graph to a json-like dictionary."""
    return json_graph.node_link_data(graph)


def json_to_nxgraph(jsgraph):
    """Converts a json-like dictionary to a networkx graph."""
    return json_graph.node_link_graph(jsgraph)


def molecule_to_json_string(mol):
    """Converts an rdkit molecule to a json string."""
    graph = molecule_to_nxgraph(mol)
    json_dict = nxgraph_to_json(graph)
    json_string = json_to_string(json_dict)
    return json_string


def json_string_to_molecule(json_string):
    """Converts a json string to an rdkit molecule."""
    json_dict = load_json_from_string(json_string)
    graph = json_to_nxgraph(json_dict)
    mol = nxgraph_to_molecule(graph)
    return mol
