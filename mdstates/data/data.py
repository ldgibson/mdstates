from ..util import loadfile


__all__ = ['bonds', 'radii']

bonds = loadfile('bond_distances.csv')
bonds.set_index('pair', drop=True, inplace=True)

radii = loadfile('covalent_radii.csv', skiprows=1,
                 names=['number', 'element', 'single', 'double', 'triple'])
radii.set_index('element', inplace=True)
