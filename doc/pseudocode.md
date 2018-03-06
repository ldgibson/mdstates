# Pseudo-code for `mdreactions`

1. read in trajectory and topology files [+ PBC (default=True) + bond cutoff distance]
2. generate list of atom pairs to pass to `compute_distance()` from the number of atoms in trajectory
    - only generate unique combinations of atom pairs ${n}\choose{2}$
3. compute the distances between all atom pairs generated
    - spits out a linear array of all the interatomic distances for each frame
4. change the shape of the linear arrays at each frame to an `n x n` matrix
5. determine connectivity using `cutoff` for all pairs

**Contact matrix is now generated at all frames**

6. check the contact matrix to see if there are any indices that never change
    - copy those indices into an `ignore_list`
7. on the remaining indices in the contact matrix run the Viterbi algorithm to determine most likely trajectory
    - set all indices in `ignore_list` to their respective continuous values (e.g. `0` or `1` at all frames)
8. loop through the frames in the contact matrix to check if the current frame differs from the previous frame
    - if there is a change in the contact matrix, add the frame number to a list
    - return list of frames where reactions likely occurred
9. determine the structure (e.g. SMILES) at each block of the trajectory between reactions


### Desired Functionality:
```
>>> import mdreactions as mdr

>>> CUTOFF = 0.180  # nm

>>> rnet = mdr.ReactionNetwork('trajectory.xyz', 'topology.pdb',
...                            CUTOFF, periodic=True)

>>> rnet.clean_traj()

>>> print(rnet.rxn_frames)
[351, 1278, 3020]

>>> rnet.build_network()
```
