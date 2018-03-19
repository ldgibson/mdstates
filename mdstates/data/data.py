from ..util import loadfile


__all__ = ['bonds']

bonds = loadfile('bond_distances.csv')
bonds.set_index('pair', drop=True, inplace=True)
