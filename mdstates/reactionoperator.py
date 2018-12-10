import re

import numpy as np
import pandas as pd


class AtomList:
    def __init__(self, atoms, _df=None):
        if _df is None:
            self.df = pd.DataFrame({'symbol': atoms,
                                    'label': self._enumerate_atoms(atoms),
                                    'index': range(len(atoms))})
        else:
            self.df = _df
        self.symbols = self.df.loc[:, 'symbol'].values
        self.labels = self.df.loc[:, 'label'].values
        return

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __setitem__(self, idx, value):
        self.df.iloc[idx] = value

    def __delitem__(self, idx):
        self.df = self.df.drop(index=idx)

    def __iter__(self):
        for idx, atom in self.df.iterrows():
            yield atom

    def __str__(self):
        string = ''
        for atom in self.__iter__():
            string += '{} '.format(atom.label)
        string += '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, AtomList):
            return self.labels == other.labels
        else:
            raise TypeError("Both objects just be AtomList objects.")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.df)

    def _enumerate_atoms(self, atoms_list):
        atoms_array = np.array(atoms_list, dtype="U8")
        count_dict = {atom: 0 for atom in np.unique(atoms_array)}
        for i, atom in enumerate(atoms_array):
            count_dict[atom] += 1
            atoms_array[i] = atom + str(count_dict[atom])
        return atoms_array


class BEMatrix(np.ndarray):
    def __new__(cls, a, atoms=None):
        obj = np.asarray(a).view(cls)
        # if np.all(np.tril(obj, -1) == 0):
            # obj = obj + np.transpose(obj, [1, 0])
        # else:
            # pass

        obj.atoms = atoms
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.atoms = getattr(obj, 'atoms', None)

    def __getitem__(self, idx):
        new = super().__getitem__(idx)
        if isinstance(idx, tuple):
            if isinstance(idx[0], slice):
                new_atoms = self.atoms[idx[0]]
            elif isinstance(idx[0], list):
                new_atoms = [self.atoms[i] for i in idx[0]]
            elif isinstance(idx[0], int):
                new_atoms = self.atoms[idx[0]]
            else:
                raise TypeError("Must passing proper index values.")
        else:
            new_atoms = self.atoms[idx]
        new_mat = BEMatrix(new, atoms=new_atoms)
        return new_mat
    
    def __setitem__(self, idx, value):
        super().__setitem__(idx, value)
    
    def __delitem__(self, idx):
        raise Exception("Cannot delete items in this way. Use ",
                        "BEMatrix().delete(...) instead.")
    
    def __str__(self):
        arr = self.view(np.ndarray)
        if len(arr.shape) == 1:
            arr = arr.reshape(arr.shape[0], -1)
        else:
            pass
        i_indices = range(arr.shape[0])
        j_indices = range(1, arr.shape[1])

        print_string = ""

        for i in i_indices:
            print_string += "{:<3} [".format(self.atoms[i])
            print_string += "{:>-2d}".format(arr[i, 0])
            if j_indices is None:
                pass
            else:
                for j in j_indices:
                    print_string += "{:>-3d}".format(arr[i, j])
            print_string += "]"
            if i != arr.shape[0] - 1:
                print_string += "\n"
        return print_string
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if not isinstance(other, BEMatrix):
            raise TypeError("Both objects must be BEMatrix objects.")
        else:
            pass
        
        if self.atoms == other.atoms:
            return super().__add__(other)
        else:
            raise ValueError("Both objects must have the same atom list.")
    
    def __sub__(self, other):
        if not isinstance(other, BEMatrix):
            raise TypeError("Both objects must be BEMatrix objects.")
        else:
            pass
        
        if self.atoms == other.atoms:
            return super().__sub__(other)
        else:
            raise ValueError("Both objects must have the same atom list.")
    
    def is_equal(self, other):
        if isinstance(other, BEMatrix):
            for at1, at2 in zip(self.atoms, other.atoms):
                if re.search("\d", at1):
                    atom1 = re.search("[a-zA-Z]*", at1).group(0)
                else:
                    atom1 = at1
                if re.search("\d", at2):
                    atom2 = re.search("[a-zA-Z]*", at2).group(0)
                else:
                    atom2 = at2
                if atom1 != atom2:
                    return False
            return super().__eq__(other)
        else:
            raise Exception("Both objects must BEMatrix objects.")

    # def __eq__(self, other):
        # if isinstance(other, BEMatrix):
            # for at1, at2 in zip(self.atoms, other.atoms):
                # if re.search("\d", at1):
                    # atom1 = re.search("\d", at1).group(0)
                # else:
                    # atom1 = at1
                # if re.search("\d", at2):
                    # atom2 = re.search("\d", at2).group(0)
                # else:
                    # atom2 = at2
                # if atom1 != atom2:
                    # raise Exception("Atom lists of the two objects "
                                    # "do not match.")
            # return super().__eq__(other)
        # else:
            # raise Exception("Both objects must BEMatrix objects.")

    # def __ne__(self, other):
        # return super().__eq__(other)
    
    def delete(self, idx):
        cpy = self.view(np.ndarray)
        del_rows = np.delete(cpy, idx, 0)
        new = np.delete(del_rows, idx, 1)
        new_atoms = self.atoms.copy()
        if isinstance(idx, list):
            for i in sorted(idx, reverse=True):
                del new_atoms[i]
        elif isinstance(idx, int):
            del new_atoms[idx]
        else:
            raise TypeError('Must pass either an int or list of int.')
        new_bematrix = BEMatrix(new, atoms=new_atoms)
        return new_bematrix
    
    @property
    def atoms(self):
        return self.__atoms

    @atoms.setter
    def atoms(self, values):
        if values is None:
            self.__atoms = ["XX".format(i) for i in range(self.shape[0])]
        else:
            self.__atoms = values
        return


# class BEMatrix:
    # def __init__(self, array, atoms=None):
        # if np.all(np.tril(array, -1) == 0):
            # array = array + np.transpose(array, [1, 0])
        # else:
            # pass

        # self.array = array
        # self.atoms = atoms
        # return

    # def __str__(self):
        # print_string = ""
        # if self.array.shape[0] > 10:
            # i_indices = [0, 1, 2, 3, -4, -3, -2, -1]
            # j_indices = [1, 2, 3, -4, -3, -2, -1]
        # else:
            # i_indices = range(self.array.shape[0])
            # j_indices = range(1, self.array.shape[1])

        # for i in i_indices:
            # if i == -4:
                # print_string += "..."
                # for _ in range(30):
                    # print_string += " "
                # print_string += "\n"
            # else:
                # pass

            # print_string += "{:<3} [".format(self.atoms[i].label)
            # print_string += "{:>-2d}".format(self.array[i, 0])

            # for j in j_indices:
                # if j == -4:
                    # print_string += " ..."
                # else:
                    # pass
                # print_string += "{:>-3d}".format(self.array[i, j])
            # print_string += "]"
            # if i != self.array.shape[0] - 1:
                # print_string += "\n"
        # return print_string

    # def __repr__(self):
        # return self.__str__()

    # def __getitem__(self, idx):
        # new_mat = BEMatrix(self.array[idx, idx], self.atoms[idx].symbol.tolist())
        # return new_mat

    # def __setitem__(self, idx, val):
        # self.array[idx, idx] = val
        # return

    # def __delitem__(self, idx):
        # if isinstance(idx, slice):
            # if idx.step is None:
                # idx = list(range(idx.start, idx.stop))
            # else:
                # idx = list(range(idx.start, idx.stop, idx.step))
        # else:
            # pass
        # rows_deleted = np.delete(self.array, idx, 0)
        # op = np.delete(rows_deleted, idx, 1)
        # del self.array[idx, idx]

    # def __eq__(self, other):
        # if isinstance(other, BEMatrix):
            # if np.all(other.atoms == self.atoms):
                # return self.array == other.array
            # else:
                # raise Exception("Atom lists of the two objects do not match.")
        # else:
            # raise Exception("Both objects must BEMatrix objects.")

    # def __ne__(self, other):
        # return self.__eq__(other)

    # def __sub__(self, other):
        # assert np.all(self.atoms == other.atoms),\
            # "Both objects must be BEMatrix objects."
        # result = BEMatrix(self.array - other.array, self.atoms)
        # return result

    # @property
    # def atoms(self):
        # return self.__atoms

    # @atoms.setter
    # def atoms(self, values):
        # if values is None:
            # self.__atoms = AtomList(np.array(["XX".format(i) for i in
                                              # range(self.array.shape[0])]))
        # else:
            # if isinstance(values, AtomList):
                # self.__atoms = values
                # self.atoms.df.index = range(len(values.df))
            # else:
                # self.__atoms = AtomList(values)
        # return


# class ReactionOperator:
    # def __init__(self, reactant, product, atoms=None):
        # assert reactant.shape[0] == reactant.shape[1],\
            # "Reactant array must be square."
        # assert product.shape[0] == product.shape[1],\
            # "Product array must be square."

        # # if isinstance(atoms, np.ndarray):
            # # if atoms.dtype == "U8":
                # # pass
            # # else:
                # # atoms.dtype = "U8"
        # # else:
            # # atoms = np.array(atoms, dtype="U8")

        # # if atoms is None:
            # # self.atoms = AtomList(np.array(["XX".format(i) for i in
                                            # # range(reactant.array.shape[0])]))
        # # else:
            # # if isinstance(atoms, AtomList):
                # # self.atoms = atoms
            # # else:
                # # self.atoms = AtomList(atoms)

        # if atoms is None or isinstance(atoms, list):
            # self.atoms = atoms
        # else:
            # raise TypeError("Must pass a list of strings, or None.")

        # if not isinstance(reactant, BEMatrix):
            # self.reactant = BEMatrix(reactant, atoms=self.atoms)
        # else:
            # self.reactant = reactant
            # self.reactant.atoms = self.atoms
        # if not isinstance(product, BEMatrix):
            # self.product = BEMatrix(product, self.atoms)
        # else:
            # self.product = product
            # self.reactant.atoms = self.atoms

        # self.operator = self._get_reaction_operator(self.reactant,
                                                    # self.product)
        # return

    # def __str__(self):
        # return self.operator.__str__()

    # def __repr__(self):
        # return self.operator.__str__()

    # def show_reactant_to_product(self):
        # react_str = self.reactant.__str__().split('\n')
        # prod_str = self.product.__str__().split('\n')
        # join_str = "     "
        # join_str_arrow = " --> "
        # if len(react_str) % 2 == 0:
            # arrow_line = len(react_str) // 2 - 1
        # else:
            # arrow_line = len(react_str) // 2
        # joined = []
        # for i, (line1, line2) in enumerate(zip(react_str, prod_str)):
            # if i != arrow_line:
                # joined.append(join_str.join([line1, line2]))
            # else:
                # joined.append(join_str_arrow.join([line1, line2]))
        # print('\n'.join(joined))
        # # return '\n'.join(joined)
        # return

    # def _enumerate_atoms(self, atoms_list):
        # atoms_array = np.array(atoms_list, dtype="U8")
        # count_dict = {atom: 0 for atom in np.unique(atoms_array)}
        # for i, atom in enumerate(atoms_array):
            # count_dict[atom] += 1
            # atoms_array[i] = atom + str(count_dict[atom])
        # return atoms_array

    # def _get_reaction_operator(self, reactant, product):
        # diff = product - reactant
        # zero_indices = self._get_zero_columns(diff)
        # smaller = diff.delete(zero_indices)
        # # nonzero_indices = np.array([idx for idx in range(len(diff))
                                    # # if idx not in zero_indices])
        # # rows_deleted = np.delete(diff, zero_indices, 0)
        # # op = np.delete(rows_deleted, zero_indices, 1)
        # # bematrix_op = BEMatrix(op)
        # # reaction_operator_atoms = reactant.atoms[nonzero_indices]
        # # reaction_operator_atoms.index = range(len(reactant.atoms[nonzero_indices]))
        # # bematrix_op.atoms = AtomList(None, reaction_operator_atoms)
        # return smaller

    # def _get_zero_columns(self, arr):
        # indices = []
        # for i in range(arr.shape[1]):
            # if np.all(arr[:, i] == 0):
                # indices.append(i)
            # else:
                # pass
        # return indices


class ReactionOperator:
    def __init__(self, reactant, product):
        if not isinstance(reactant, BEMatrix):
            raise TypeError("Reactant argument must be a BEMatrix object.")
        else:
            if np.all(np.tril(reactant, -1) == 0):
                reactant = reactant + np.transpose(reactant, [1, 0])
            else:
                pass
            self.reactant = reactant

        if not isinstance(product, BEMatrix):
            raise TypeError("Product argument must be a BEMatrix object.")
        else:
            if np.all(np.tril(product, -1) == 0):
                product = product + np.transpose(product, [1, 0])
            else:
                pass
            self.product = product

        assert reactant.shape[0] == reactant.shape[1],\
            "Reactant array must be square."
        assert product.shape[0] == product.shape[1],\
            "Product array must be square."
        
        assert product.atoms == reactant.atoms,\
            "Atom lists are not the same between reactants "

        self.atoms = reactant.atoms
        self.operator = self._get_reaction_operator(self.reactant,
                                                    self.product)
        return
    
    def __str__(self):
        return self.operator.__str__()

    def __repr__(self):
        return self.operator.__str__()
    
    def __getitem__(self, idx):
        return self.operator[idx]

    def show_reactant_to_product(self):
        react_str = self.reactant.__str__().split('\n')
        prod_str = self.product.__str__().split('\n')
        join_str = "     "
        join_str_arrow = " --> "
        if len(react_str) % 2 == 0:
            arrow_line = len(react_str) // 2 - 1
        else:
            arrow_line = len(react_str) // 2
        joined = []
        for i, (line1, line2) in enumerate(zip(react_str, prod_str)):
            if i != arrow_line:
                joined.append(join_str.join([line1, line2]))
            else:
                joined.append(join_str_arrow.join([line1, line2]))
        print('\n'.join(joined))
        return

    def _enumerate_atoms(self, atoms_list):
        atoms_array = np.array(atoms_list, dtype="U8")
        count_dict = {atom: 0 for atom in np.unique(atoms_array)}
        for i, atom in enumerate(atoms_array):
            count_dict[atom] += 1
            atoms_array[i] = atom + str(count_dict[atom])
        return atoms_array
    
    def _get_reaction_operator(self, reactant, product):
        diff = product - reactant
        zero_indices = self._get_zero_columns(diff)
        smaller = diff.delete(zero_indices)
        return smaller

    def _get_zero_columns(self, arr):
        indices = []
        for i in range(arr.shape[1]):
            if np.all(arr[:, i] == 0):
                indices.append(i)
            else:
                pass
        return indices


def sort_BEMatrix(mat, sort_priority):
    """Uses the sorting priority to return a sorted BEMatrix.

    Parameters
    ----------
    mat : BEMatrix
    sort_priority : dict
        Dictionary with unique atom types as keys with their values as positive
        integers that represent the sorting priority. Higher values are placed
        earlier.

    Returns
    -------
    BEMatrix
    """
    values = np.unique([sort_priority[at[:-1]] for at in mat.atoms])
    values = sorted(values, reverse=True)
    inv_dict = {v: k for k, v in sort_priority.items()}
    new_atom_order = [inv_dict[val] for val in values]
    old_idx = []
    for at1 in new_atom_order:
        for i, at2 in enumerate(mat.atoms):
            if at2[:-1] == at1:
                old_idx.append(i)
    new = mat[old_idx, :]
    new2 = new[:, old_idx]
    return new2


def get_unique(matrix_list):
    unique = []

    for i, op in enumerate(matrix_list):
        sorted_operator = sort_BEMatrix(op, sort_order)
        if unique:
            for op2 in unique:
                if np.all(sorted_operator.is_equal(op2)):
                    break
                else:
                    pass
            else:
                unique.append(sorted_operator)
        else:
            unique.append(sorted_operator)
    return unique
