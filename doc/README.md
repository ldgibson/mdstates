# Use cases
1. Analyze MD trajectories in which the connectivity (bonded/nonbonded) of the system changes and locate the frames at which transitions occur.
2. Generate a reaction network graph that displays the various states visted by the system.
    - Generates PDB files for each state and saves them into a `structures/` directory in the current working directory.
3. For each transition detected, generate CP2K input files for NEB calculations. A frame before, at, and after each transition is used to guide the NEB calculation.
    - User can choose:
        - level of theory
        - number of frames from the transition when grabbing frames before and after the transition (i.e. how many frames +/- the transition)
