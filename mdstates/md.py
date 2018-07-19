"""
Workflow:
 1. Get list of nodes.
 2. Specify level of theory and duration of run.
 3. Generate directories for each node.
    a. Generate input structures of each node.
    b. Copy input file for CP2K into each directory.
 4. Run each job the duration of the time.
 5. Get all energies of each job and perform Boltzmann averaging.
 6. Return a dictionary containing SMILES as keys and average
    energies as values.
"""

import ase.io
from ase import Atoms, units
from ase.calculators.cp2k import CP2K
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


CP2K.command = "mpirun -n 1 -genv OMP_NUM_THREADS 1 /gscratch/pfaendtner/codes/cp2k/cp2k/cp2k/exe/Linux-x86-64-intel/cp2k_shell.psmp"


_inp_pm6="""
&FORCE_EVAL
  ! the electronic structure part of CP2K is named Quickstep
  METHOD Quickstep
  &DFT
    ! basis sets and pseudopotential files can be found in cp2k/data

    ! Charge and multiplicity

    &QS
       ! use the GPW method (i.e. pseudopotential based calculations with the Gaussian and Plane Waves scheme).
       METHOD PM6
       ! default threshold for numerics ~ roughly numerical accuracy of the total energy per electron,
       ! sets reasonable values for all other thresholds.
       EPS_DEFAULT 1.0E-10 
       ! used for MD, the method used to generate the initial guess.
       EXTRAPOLATION ASPC 
    &END

    &POISSON
       PERIODIC XYZ ! the default, gas phase systems should have 'NONE' and a wavelet solver
    &END

    &PRINT
       ! at the end of the SCF procedure generate cube files of the density
       &E_DENSITY_CUBE OFF
       &END E_DENSITY_CUBE
    &END

    ! use the OT METHOD for robust and efficient SCF, suitable for all non-metallic systems.
    &SCF                              
      SCF_GUESS ATOMIC ! can be used to RESTART an interrupted calculation
      MAX_SCF 20
      EPS_SCF 1.0E-6 ! accuracy of the SCF procedure typically 1.0E-6 - 1.0E-7
      &OT
        ! an accurate preconditioner suitable also for larger systems
        PRECONDITIONER FULL_SINGLE_INVERSE
        ! the most robust choice (DIIS might sometimes be faster, but not as stable).
        MINIMIZER DIIS
      &END OT
      &OUTER_SCF ! repeat the inner SCF cycle 10 times
        MAX_SCF 20
        EPS_SCF 1.0E-6 ! must match the above
      &END
      &PRINT
          &RESTART OFF
          &END
      &END
    &END SCF

  &END DFT
&END FORCE_EVAL
"""


def get_smiles():

    return


def run_md(starting_structure, dt=1.0, T=300, steps=1000, write_interval=10,
           ase_traj='md.traj', xyz_name='md.xyz', cell=[10, 10, 10], pbc=True,
           inp=_inp_pm6):
    """Runs NVT molecular dynamics on the starting structure.

    Parameters
    ----------
    starting_structure : str
        Input file name of the starting structure.
    dt : int or float, optional
        Time step.
    T : int or float, optional
        Temperature of the system.
    steps : int, optional
        Number of MD steps.
    write_interval : int, optional
        The rate at which output to trajectory files is written.
    ase_traj : str, optional
        Name of ASE trajectory file.
    xyz_name : str, optional
        Name of XYZ trajectory file.
    cell : list of int or float, optional
        Cell parameters of the system.
    pbc : bool, optional
        Sets periodic boundary condition.
    inp : str, optional
        Input file snippet for CP2K. Default is PM6 snippet.

    """
    calc = CP2K(xc=None,
                basis_set=None,
                basis_set_file=None,
                cutoff=None,
                max_scf=None,
                force_eval_method=None,
                potential_file=None,
                poisson_solver=None,
                pseudo_potential=None,
                stess_tensor=False,
                inp=_inp_pm6)
    traj = ase.io.Trajectory(ase_traj, 'w')

    atoms = ase.io.read(starting_structure)
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)
    atoms.set_calculator(calc=calc)
    atoms.get_potential_energy()
    traj.write(atoms)
    ase.io.write(xyz_name, atoms)

    MaxwellBoltzmannDistribution(atoms, T * units.kB)
    dyn = VelocityVerlet(atoms, dt=dt * units.fs)

    # Steps for for-loop, uses integer division '//' to prevent
    # float quantities.
    num_cycles = steps // write_interval
    for step in range(num_cycles):
        dyn.run(write_interval)
        atoms.get_potential_energy()
        traj.write(atoms)
        ase.io.write(xyz_name, atoms, append=True)

    return


def boltzmann_average_energies():

    return


def get_node_energies(network, steps=10000, T=300, xc='PM6',
                      basis='6-311G**', cell=[]):
    """Calculates energies of each node in network, if stable.

    Runs molecular dynamics of each structure and calculates the
    Boltzmann average of the energies across the simulation. This is
    used to get a more accurate representation of the energy of
    multi-component systems, i.e. systems with more than a single
    molecule.

    Parameters
    ----------
    network : networkx.Graph
        Reaction network graph containing SMILES as nodes.
    steps : int, optional
        Number of steps to run in molecular dynamics.
    T : int or float, optional
        Temperature of the simulation.
    xc : str, optional
        Exchange-correlation functional. Currently only accepts `B3PW91`
        or `PM6`
    basis : str, optional
        Basis set. Currently only taking Pople-type basis sets. If `xc`
        is set to `PM6`, then this argument is ignored.
    cell : list of int or float, optional
        Sets the simulation box size if specified.
    
    Returns
    -------
    node_energies : dict
        Dictionary containing SMILES as keys and energies as values.
        Energy units are kcal/mol.
    """

    return
