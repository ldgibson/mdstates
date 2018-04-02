import re
import os.path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions

# __all__ = ['issameSMILES', 'isinSMILESlist', 'differentstrings',
#           'replaceSMILES', 'swapconformerSMILES', 'reduceSMILES',
#           'uniqueSMILES', 'SMILEStofile', 'SMILESlisttofile',
#           'save_unique_SMILES', 'SMILESfingerprint']


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
        elif smi == smiles_list[i - 1]:
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
        if smi not in unique:
            unique.append(smi)
    return unique


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


def saveSMILESimages(smiles_list, location="SMILESimages", size=(400, 400),
                     bondlinewidth=2.0, atomlabelfontsize=16, fit_image=True,
                     rewrite=False):
    """Saves images of a list of SMILES strings to a specified folder.

    Parameters
    ----------
    smiles_list : list of str
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
        os.system("rm -rf " + location)
        os.system("mkdir " + location)
    else:
        if not os.path.exists(location):
            os.system("mkdir " + location)
        else:
            pass

    DrawingOptions.bondLineWidth = bondlinewidth
    DrawingOptions.atomLabelFontSize = atomlabelfontsize

    for smi in smiles_list:
        if os.path.exists(os.path.join(location, smi + '.png')):
            pass
        else:
            SMILEStofile(smi, os.path.join(location, smi + '.png'), fit_image,
                         size=size)
    return


def save_unique_SMILES(smiles_list):
    unique = uniqueSMILES(smiles_list)
    saveSMILESimages(unique)
    return


def _break_ionic_bonds(smiles):
    if isinstance(smiles, str):
        return re.sub("\[Li\]O", "[Li].[O]", smiles)
    elif isinstance(smiles, list):
        return [re.sub("\[Li\]O", "[Li].[O]", smi) for smi in smiles]


def _radical_to_sp2(smiles):
    if isinstance(smiles, str):
        return re.sub("\[C\]\(\[O\]\)", "C(=O)", smiles)
    elif isinstance(smiles, list):
        return [re.sub("\[C\]\(\[O\]\)", "C(=O)", smi) for smi in smiles]
