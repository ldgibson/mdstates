import os
import os.path 
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem.Fingerprints import FingerprintMols

__all__ = ['issameSMILES', 'isinSMILESlist', 'differentstrings',
           'replaceSMILES', 'swapconformerSMILES', 'reduceSMILES',
           'uniqueSMILES', 'SMILEStofile', 'SMILESlisttofile']


def issameSMILES(smiles1, smiles2):
    """Checks if two SMILES strings match.

    Generates chemical fingerprints for every SMILES string and
    evaluates the fingerprint similarities between `smiles` and
    every element in `smiles_list` using the Tanimoto metric.
    If any of the similarities are perfect matches, then this
    function returns `True`, otherwise `False`.

    Parameters
    ----------
    smiles1, smiles2 : str

    Returns
    -------
    bool
        Returns `False` if `smiles` does not match any SMILES strings in
        `smiles_list`, or if `smiles_list` is empty. Returns `True` if
        `smiles` does match at least one SMILES string in `smiles_list`.

    Raises
    ------
    TypeError
        If `smiles1` or `smiles2` are not strings.
    SyntaxError
        If `smiles1` or `smiles2` are empty.
    SyntaxError
        If any of the SMILES strings passed are not valid.
    """
    if not isinstance(smiles1, str) or not isinstance(smiles2, str):
        raise TypeError("Must pass strings.")
    else:
        pass

    if not smiles1 or not smiles2:
        raise SyntaxError("Cannot pass an empty string.")
    else:
        pass

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None:
        raise SyntaxError("{} is not a valid SMILES string.".format(smiles1))
    elif mol2 is None:
        raise SyntaxError("{} is not a valid SMILES string.".format(smiles2))
    else:
        pass

    if smiles1 == smiles2:
        return True
    else:
        pass

    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)

    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)

    if np.isclose(similarity, 1):
        return True
    else:
        return False


def isinSMILESlist(smiles, smiles_list, tuples=False):
    """Checks if a SMILES string matches any SMILES in a list.

    Generates chemical fingerprints for every SMILES string and evaluates the
    fingerprint similarities between `smiles` and every element in
    `smiles_list` using the Tanimoto metric.  If any of the similarities are
    perfect matches, then this function returns `True`, otherwise `False`.

    Parameters
    ----------
    smiles_list : list of str or list of tuple of str
        List containing SMILES strings.
    smiles : str

    Returns
    -------
    bool
        Returns `False` if `smiles` does not match any SMILES strings in
        `smiles_list`, or if `smiles_list` is empty. Returns `True` if `smiles`
        does match at least one SMILES string in `smiles_list`.

    Raises
    ------
    TypeError
        If `smiles_list` is not a list of strings.
    TypeError
        If `smiles` is not a string.
    SyntaxError
        If `smiles` is empty.
    SyntaxError
        If any of the SMILES strings passed are not valid.
    """

    if not tuples:
        if not smiles:
            raise SyntaxError("Cannot pass an empty string.")
        else:
            pass

        if not isinstance(smiles_list, list):
            raise TypeError("`smiles_list` must be of `list` type.")
        else:
            pass

        is_string = [isinstance(smi, str) for smi in smiles_list]
        if not all(is_string):
            raise TypeError("All items in `smiles_list` must be strings.")
        else:
            pass

        if Chem.MolFromSmiles(smiles) is None:
            raise SyntaxError("{} : invalid SMILES string.".format(smiles))
        else:
            pass
    else:
        if not smiles[0] or not smiles[1]:
            raise SyntaxError("Cannot pass an empty string.")
        elif not isinstance(smiles[0], str) or\
                not isinstance(smiles[1], str):
            raise TypeError("SMILES must be strings.")
        else:
            pass

        is_string = [all((isinstance(smi1, str),
                          isinstance(smi2, str)))
                     for smi1, smi2 in smiles_list]
        if not all(is_string):
            raise TypeError("All items in `smiles_list` must be strings.")
        else:
            pass

        mol1 = Chem.MolFromSmiles(smiles[0])
        mol2 = Chem.MolFromSmiles(smiles[1])

        if mol1 is None:
            raise SyntaxError("{} : invalid SMILES string.".format(smiles[0]))
        elif mol2 is None:
            raise SyntaxError("{} : invalid SMILES string.".format(smiles[1]))
        else:
            pass

    if not smiles_list:
        return False
    else:
        if not tuples:
            for smi in smiles_list:
                if Chem.MolFromSmiles(smi) is None:
                    raise SyntaxError("{} : ".format(smi) +
                                      "invalid SMILES string.")
                else:
                    pass
        else:
            for smi1, smi2 in smiles_list:
                mol1 = Chem.MolFromSmiles(smi1)
                mol2 = Chem.MolFromSmiles(smi2)

                if mol1 is None:
                    raise SyntaxError("{} : ".format(smi1) +
                                      "invalid SMILES string.")
                elif mol2 is None:
                    raise SyntaxError("{} : ".format(smi2) +
                                      "invalid SMILES string.")
                else:
                    pass

    if not tuples:
        similarity = [issameSMILES(smiles, smi) for smi in smiles_list]
    else:
        similarity = [all((issameSMILES(smiles[0], smi1),
                           issameSMILES(smiles[1], smi2)))
                      for smi1, smi2 in smiles_list]

    if any(similarity):
        return True
    else:
        return False


def differentstrings(smiles1, smiles2):
    """
    Checks if 2 SMILES strings have same fingerprint but different strings.
    """

    if smiles1 != smiles2 and issameSMILES(smiles1, smiles2):
        return True
    else:
        return False


def replaceSMILES(smiles_list, i, smiles):
    """Replaces a list entry with another entry.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings.
    i : int
        Index of entry to be replaced.
    smiles: str
        SMILES string to replace list entry."""

    del smiles_list[i]
    smiles_list.insert(i, smiles)
    return


def swapconformerSMILES(smiles_list, inplace=True):
    """Replaces SMILES strings with equivalent SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings.
    inplace : bool, optional
        If `True`, replace elements in original list and return `None`.
        Otherwise, copy the original list, modify it, and return it. Default
        is `True`.

    Returns
    -------
    smiles_list : list of str
        Copy of the original argument with replacements made."""

    if not inplace:
        smiles_list = smiles_list.copy()
    else:
        pass

    for i in range(1, len(smiles_list)):
        for j in range(len(smiles_list[:i-1])):
            if differentstrings(smiles_list[i], smiles_list[j]):
                replaceSMILES(smiles_list, i, smiles_list[j])
            else:
                pass

    if not inplace:
        return smiles_list
    else:
        return

def reduceSMILES(smiles_list):
    """Removes repeated SMILES strings in a list.

    Parameters
    ----------
    smiles_list : list of str
    
    Returns
    -------
    reduced_smiles : list of str
        List of SMILES strings with consecutive repeats removed.

    Example
    -------
    >>> smiles = ['C(=O)=O', 'C(=O)=O', 'C(=O)=O', 'O=O', 'O=O']
    >>> reduceSMILES(smiles)
    ['C(=O)=O', 'O=O']
    """
    reduced_smiles = []

    for i, smi in enumerate(smiles_list):
        if i == 0:
            reduced_smiles.append(smi)
        elif issameSMILES(smi, smiles_list[i-1]):
            pass
        else:
            reduced_smiles.append(smi)

    return reduced_smiles

def uniqueSMILES(smiles_list):
    """Finds all unique SMILES in a list.

    Compares SMILES string to all SMILES strings in `unique` and only
    appends it if the fingerprint does not match any of the
    fingerprints in `unique`.

    Parameters
    ----------
    smiles_list : list of str

    Returns
    -------
    unique : list of str
        All unique SMILES strings in `smiles_list`.
    """
    unique = []

    for smi in smiles_list:
        if not isinSMILESlist(smi, unique):
            unique.append(smi)
    return unique

def SMILEStofile(smiles, filename, fit_image, show=False):
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
    Draw.MolToFile(mol, filename, fitImage=fit_image)
    if show:
        Draw.MolToImage(mol)
    return

def SMILESlisttofile(smiles_list, location="SMILESimages", bondlinewidth=2.0,
                     atomlabelfontsize=16, fit_image=True):
    """Saves images of a list of SMILES strings to a specified folder.

    Parameters
    ----------
    smiles_list : list of str
    location : str
        Folder name to save all generated images into.
    bondlinewidth : float or int
        Controls width of bond lines when drawing 2D structures of
        SMILES strings.
    atomlabelfontsize : float or int
        Controls the font size of all atom labels when drawing 2D
        structures of SMILES strings.
    """

    os.system("rm -rf " + location)
    os.system("mkdir " + location)

    DrawingOptions.bondLineWidth = bondlinewidth
    DrawingOptions.atomLabelFontSize = atomlabelfontsize

    for smi in smiles_list:
        SMILEStofile(smi, os.path.join(location,smi+'.png'), fit_image)
    return
