# mdstates
[![Build Status](https://travis-ci.org/ldgibson/mdstates.png?branch=master)](https://travis-ci.org/ldgibson/mdstates)
[![Coverage Status](https://coveralls.io/repos/github/ldgibson/mdstates/badge.svg?branch=master)](https://coveralls.io/github/ldgibson/mdstates?branch=master)

MD-States is a python package for the analysis and visualization of networks (e.g. chemical reaction networks) from molecular dynamics (MD) trajectories. This package is developed and maintained by members of the [Pfaendtner Research Group](http://prg.washington.edu/) in the Department of Chemical Engineering at the University of Washington in Seattle.

Documentation coming soon.

## Python Dependencies
  - python>=3.6
  - mdtraj
  - rdkit
  - numpy
  - pandas
  - networkx>=2.1
  - pygraphviz
  - aggdraw

## Installation
```
$ git clone github.com/ldgibson/mdstates.git
$ cd mdstates
$ conda env create  # Make sure the 'environment.yml' is present.
$ source activate mdstates
$ python setup.py install
```
