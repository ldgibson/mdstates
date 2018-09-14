# mdstates
[![Build Status](https://travis-ci.org/ldgibson/mdstates.png?branch=master)](https://travis-ci.org/ldgibson/mdstates)
[![Coverage Status](https://coveralls.io/repos/github/ldgibson/mdstates/badge.svg?branch=master)](https://coveralls.io/github/ldgibson/mdstates?branch=master)

MD-States is a python package for the analysis and visualization of networks (e.g. chemical reaction networks) from molecular dynamics (MD) trajectories based on the work done by Wang et al. (doi: [10.1021/acs.jctc.5b00830](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00830)). This package is developed and maintained by members of the [Pfaendtner Research Group](http://prg.washington.edu/) in the Department of Chemical Engineering at the University of Washington in Seattle.

Documentation coming soon.

## Python Dependencies
  - python (>=3.6)
  - mdtraj (>=1.9.1)
  - rdkit (>=2017.09.3)
  - networkx (>=2.1)
  - graphviz (>=2.38)
  - pygraphviz (>=1.4rc1)
  - aggdraw (>=1.3.4.dev0)

## Installation
```
$ git clone https://github.com/ldgibson/mdstates.git
$ cd mdstates
$ conda env create  # Make sure the 'environment.yml' is present.
$ source activate mdstates
$ python setup.py install
```
