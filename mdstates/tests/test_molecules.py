import numpy as np
from rdkit import Chem

from ..molecules import (set_structure, estimate_bonds, build_radical_graph,
                         set_positive_charges, molecule_to_contact_matrix)


def test_set_structure():
    mol = Chem.RWMol()

    cmat = np.array([[0, 0, 1],
                     [0, 0, 1],
                     [0, 0, 0]])

    atom_list = ['H', 'H', 'O']

    set_structure(mol, cmat, atom_list)

    assert mol.GetAtoms()[0].GetSymbol() == 'H', "1st atom must be hydrogen."
    assert mol.GetAtoms()[1].GetSymbol() == 'H', "2nd atom must be hydrogen."
    assert mol.GetAtoms()[2].GetSymbol() == 'O', "3rd atom must be oxygen."

    for atom in mol.GetAtoms():
        if atom.GetSymbol == 'O':
            assert atom.GetDegree() == 2, "Oxygen must only have 2 bonds."
            for n in atom.GetNeighbors():
                assert n.GetSymbol() == 'H', "Neighbors must be hydrogens."
    return


def test_build_radical_graph():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom('C'))
    mol.AddAtom(Chem.Atom('O'))
    mol.AddAtom(Chem.Atom('C'))
    mol.AddAtom(Chem.Atom('C'))

    mol.AddAtom(Chem.Atom('H'))
    mol.AddAtom(Chem.Atom('H'))

    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)

    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol.AddBond(0, 2, Chem.BondType.SINGLE)
    mol.AddBond(0, 4, Chem.BondType.SINGLE)
    mol.AddBond(2, 3, Chem.BondType.SINGLE)
    mol.AddBond(3, 5, Chem.BondType.SINGLE)

    Chem.SanitizeMol(mol)

    test_graph = build_radical_graph(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            atom_indices = [at.GetIdx() for at in test_graph]
            assert atom.GetIdx() in atom_indices,\
                "Radical atom index {} not in graph.".format(atom.GetIdx())
        else:
            pass
    return


def test_estimate_bonds():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom('C'))
    mol.AddAtom(Chem.Atom('O'))
    mol.AddAtom(Chem.Atom('C'))
    mol.AddAtom(Chem.Atom('C'))

    mol.AddAtom(Chem.Atom('H'))
    mol.AddAtom(Chem.Atom('H'))

    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)

    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol.AddBond(0, 2, Chem.BondType.SINGLE)
    mol.AddBond(0, 4, Chem.BondType.SINGLE)
    mol.AddBond(2, 3, Chem.BondType.SINGLE)
    mol.AddBond(3, 5, Chem.BondType.SINGLE)

    Chem.SanitizeMol(mol)

    # graph = build_radical_graph(mol)
    estimate_bonds(mol)

    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() ==\
        Chem.BondType.DOUBLE, "The C-O bond should be a double bond."

    assert mol.GetBondBetweenAtoms(0, 2).GetBondType() ==\
        Chem.BondType.SINGLE, "This C-C bond should be a single bond."

    assert mol.GetBondBetweenAtoms(2, 3).GetBondType() ==\
        Chem.BondType.TRIPLE, "This C-C bond should be a triple bond."
    return


def test_set_positive_charges():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom('O'))
    mol.AddAtom(Chem.Atom('H'))
    mol.AddAtom(Chem.Atom('H'))
    mol.AddAtom(Chem.Atom('H'))

    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol.AddBond(0, 2, Chem.BondType.SINGLE)
    mol.AddBond(0, 3, Chem.BondType.SINGLE)

    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)

    set_positive_charges(mol)

    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 1,\
        "Oxygen must have a positive charge."
    return


def test_molecule_to_contact_matrix():
    mol = Chem.MolFromSmiles('c1ccccc1')
    mol = Chem.AddHs(mol)
    test = molecule_to_contact_matrix(mol)
    true_atoms = np.array(['C', 'C', 'C', 'C', 'C', 'C',
                           'H', 'H', 'H', 'H', 'H', 'H'])
    assert np.all(test.atoms == true_atoms), "Incorrect atom list."
    return
