from itertools import groupby
import shutil
import os.path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions


def get_mol_dict(smiles_list):
    """
    Returns only unique species and their counts.

    Parameters
    ----------
    smiles_list : list of str
        List containing molecular species, allows for repeats.

    Returns
    -------
    list of tuple
        Each tuple contains the name of a unique molecular species
        and the number of occurrences.
    """
    if not smiles_list:
        raise Exception("No molecules found.")
    groups = groupby(smiles_list)
    return [(sum(1 for _ in group), smi) for smi, group in groups]


def to_chemical_equation(reactant_list, product_list=None):
    """
    Prints the chemical equation.

    Parameters
    ----------
    reactant_list : list of tuples
        Each tuple contains the name of a unique molecular species
        and the number of occurrences.
    product_list : list of tuples, optional
        If left empty, the function will only print the first half of the
        chemical equation.

    Returns
    -------
    str
        Either full or half of a chemical equation.
    """
    for count, mol in reactant_list:
        if not mol:
            raise Exception("Molecule missing, string is empty.")
        else:
            pass

    for count, mol in product_list:
        if not mol:
            raise Exception("Molecule missing, string is empty.")
        else:
            pass

    mol1_list = []
    for mol in reactant_list:
        if mol[0] == 1:
            mol1_list.append(mol[1])
        else:
            mol1_list.append("{} {}".format(*mol))
    if product_list:
        mol2_list = []
        for mol in product_list:
            if mol[0] == 1:
                mol2_list.append(mol[1])
            else:
                mol2_list.append("{} {}".format(*mol))

        return " + ".join(mol1_list) + " --> " + " + ".join(mol2_list)
    else:
        return " + ".join(mol1_list)


def remove_common_molecules(reactants, products):
    """
    Removes common species between two lists leaving only reacting species.

    Parameters
    ----------
    reactants, products : list of str
        List containing strings all molecular species.

    Returns
    -------
    tuple of str
        Reduced lists for both reactants and products such that only species
        that participate in the reaction remain.
    """
    reduced_react = reactants.copy()
    reduced_prod = products.copy()

    reduced_react.sort()
    reduced_prod.sort()

    if reduced_react == reduced_prod:
        raise Exception("Reactants and products are the same.")
    else:
        pass

    for mol in reactants:
        if mol in reduced_prod:
            reduced_prod.remove(mol)
            reduced_react.remove(mol)

    return (reduced_react, reduced_prod)


def find_reaction(smi1, smi2):
    """
    Writes chemical equation that shows the change from reactants to products.

    Parameters
    ----------
    smi1, smi2 : str
        SMILES strings.

    Returns
    -------
    str
        Chemical equation that contains only species that participate in the
        chemical reaction.
    """
    react = smi1.split('.')
    prod = smi2.split('.')

    reduced_react, reduced_prod = remove_common_molecules(react, prod)

    react_dict = get_mol_dict(reduced_react)
    prod_dict = get_mol_dict(reduced_prod)

    return to_chemical_equation(react_dict, prod_dict)


def remove_consecutive_repeats(smiles):
    """
    Removes consecutive repeats from a list.

    Parameters
    ----------
    smiles : list
        Raw list of SMILES strings.

    Returns
    -------
    true_list : list
        List with all consecutive repeats removed.
    """
    reduced = pd.DataFrame(columns=['smiles', 'molecule', 'frame',
                                    'transition_frame'])
    ref = smiles.loc[0, 'smiles']
    reduced = reduced.append(smiles.iloc[0], ignore_index=True)

    for i, row in smiles.iterrows():
        if i == 0:
            continue

        if row['smiles'] != ref:
            reduced = reduced.append(row, ignore_index=True)
            ref = row['smiles']
    return reduced


def uniqueSMILES(smiles_list):
    """Finds all unique SMILES in a list.

    Parameters
    ----------
    smiles_list : list of str

    Returns
    -------
    unique : list of str
        All unique SMILES strings in `smiles_list`.
    """
    return frozenset(smiles_list)


def SMILEStofile(smiles, filename, fit_image, size=(400, 400), show=False):
    """Saves image of 2D structure of SMILES string.

    Parameters
    ----------
    smiles : str
    filename : str
    show : bool
        If `True`, also print image. Otherwise, hide image and only
        save to file.
    """
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, filename, size=size, fitImage=fit_image)
    if show:
        Draw.MolToImage(mol)
    return


def saveSMILESimages(smiles, location="SMILESimages", size=(400, 400),
                     bondlinewidth=2.0, atomlabelfontsize=16, fit_image=True,
                     rewrite=False):
    """Saves images of a list of SMILES strings to a specified folder.

    Parameters
    ----------
    smiles : iterable of str
    location : str
        Folder name to save all generated images into.
    size : tuple of int
        Controls the image size in pixels.
    bondlinewidth : float or int
        Controls width of bond lines when drawing 2D structures of
        SMILES strings.
    atomlabelfontsize : float or int
        Controls the font size of all atom labels when drawing 2D
        structures of SMILES strings.
    fit_image : bool
        If `True`, an attempt is made to fill the image space with the
        image to prevent excessive white space around edges.
    rewrite : bool
        If `True`, the location of SMILES structures will be removed
        and rewritten. If `False`, files will not be overwritten if
        they already exist.
    """

    if rewrite:
        if os.path.exists(location):
            shutil.rmtree(location, ignore_errors=True)
        else:
            pass

        os.mkdir(location)
    else:
        if not os.path.exists(location):
            os.mkdir(location)
        else:
            pass

    DrawingOptions.bondLineWidth = bondlinewidth
    DrawingOptions.atomLabelFontSize = atomlabelfontsize

    for smi in smiles:
        if os.path.exists(os.path.join(location, smi + '.png')):
            pass
        else:
            SMILEStofile(smi, os.path.join(location, smi + '.png'), fit_image,
                         size=size)
    return


def save_unique_SMILES(smiles_list):
    """Finds unique SMILES and saves 2D structure images of each.

    Parameters
    ----------
    smiles_list : list of str"""

    unique = uniqueSMILES(smiles_list)
    saveSMILESimages(unique)
    return
