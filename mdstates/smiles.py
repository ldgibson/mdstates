import re
import shutil
import os.path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions


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

    print("Saving SMILES images to: {}".format(os.getcwd()))

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
