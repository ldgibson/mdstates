import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from .reactionoperator import BEMatrix


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


def build_molecule(cmat, atom_list):
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

    # Keep a copy of the molecule with explicity hydrogens
    molHs = mol

    # Remove any unnecessary hydrogens.
    mol = Chem.RemoveHs(mol)

    return mol, molHs


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

    mol = Chem.AddHs(mol)
    Chem.Kekulize(mol)
    dim = mol.GetNumAtoms(onlyExplicit=False)
    cmat = np.zeros((dim, dim), dtype=np.int32)
    atom_list = np.empty(dim, dtype='U8')
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            idx = atom.GetIntProp('index')
        except(KeyError):
            idx = i
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
        else:
            cmat[i, j] = 3
    bemat = BEMatrix(cmat, atom_list)
    return bemat
