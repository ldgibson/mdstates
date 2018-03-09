from ..util import loadfile


bonds = loadfile('bond_distances.csv')
bonds.set_index('pair', drop=True, inplace=True)
