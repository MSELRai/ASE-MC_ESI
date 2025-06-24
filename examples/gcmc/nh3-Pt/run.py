import numpy as np
from ase import units, build
from ase.io.trajectory import Trajectory
from ase.mc import BVT, Moveset, MonteCarlo
from ase.optimize.lbfgs import LBFGS

# 0) Define the potential energy surface
from mace.calculators import mace_mp
calc = mace_mp()

# 1) Find the ground state energy and geometry of an isolated NH3 molecule
probe = build.molecule("NH3")
probe.calc = calc
opt = LBFGS(probe, trajectory="lbfgs.traj", logfile="lbfgs.log")
opt.run(fmax=0.05)

# 2) Build the system we want to adsorb/desorb our probe molecule from
# Lattice constant is previously derived from an EOS fit
atoms = build.fcc111("Pt", (4,4,6), a=3.97, vacuum=15.0, orthogonal=True, periodic=True)

exclusion_list = np.arange(len(atoms)) # We don't want to translate or rotate the slab


# 3) Set up the Monte Carlo Moves
ensemble = BVT(
    temperature_K= 1000.0,
    b_parameter = [3.0],
    species = [probe],
    reference_energy=[probe.get_potential_energy()],
    rcavity = 2.5,
    grid_resolution=20,
    cavity_bias=True,
    starting_tag = 7, # Do this because build.fcc111 includes tags to identify slab layers
    exclusion_list = exclusion_list
)

moveset = Moveset(ensemble.get_moves())
moveset.adjust_parameter("Translate", "max_delta", 5.0)

# 4) Run Monte Carlo Simulation
dyn = MonteCarlo(
    atoms,
    dft_calc=calc,
    moveset=moveset,
    trajectory = "gcmc.traj",
    logfile = "gcmc.log",
    loginterval=20,
)

dyn.run(100000)

