# File for Molecule class


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
