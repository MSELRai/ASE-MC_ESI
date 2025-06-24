import sys
import numpy as np
from ase import Atoms, units
from ase.build import molecule
from ase.io.trajectory import Trajectory

from ase.calculators.lj import LennardJones

from ase.mc.moves import HMC, Insert, Delete
from ase.mc.moveset import Moveset
from ase.mc.mc import MonteCarlo
from ase.mc.utility import random_packing


B = float(sys.argv[1])
temperature_K = 120.0
probe = Atoms("Ar", [[0., 0., 0.]])
atoms = Atoms()
atoms.set_cell([20., 20., 20.])
atoms.set_pbc(True)

random_packing(atoms = atoms, adsorbate = probe, n = int(np.exp(B)), tolerance=3.0)
calc = LennardJones(sigma=3.3952, epsilon=116.79*units.kB, rc=10.0)

for atom in atoms:
    atom.tag = 0

hmc_move = HMC(
                accepted=0,
                attempted=0,
                probability=0.34,
                dt=20.0 * units.fs,
                limits=(1.0*units.fs, 50.0 * units.fs),
                nsteps=1,
                beta=1.0 / units.kB / temperature_K,
                r_overlap=0.5,
                update_frequency=1000,
                scale_delta=0.1,
                target_acceptance=0.5,
)

insert_move = Insert(
                    accepted = 0,
                    attempted = 0,
                    probability = 0.33,
                    species = probe,
                    species_tag = 0,
                    beta = 1.0 / units.kB / temperature_K,
                    b_parameter = B,
                    reference_energy = 0.0 ,
                    grid_resolution = 20,
                    rcavity = 4.0,
                    cavity_bias = True,
                    r_overlap = 0.5,
)

delete_move = Delete(
                    accepted = 0,
                    attempted = 0,
                    probability = 0.33,
                    species = probe,
                    species_tag = 0,
                    beta = 1.0 / units.kB / temperature_K,
                    b_parameter = B,
                    reference_energy = 0.0,
                    grid_resolution = 20,
                    rcavity = 4.0,
                    cavity_bias = True,
)


moveset_gcmc = Moveset([insert_move, delete_move, hmc_move])

dyn = MonteCarlo(
        atoms,
        moveset_gcmc,
        dft_calc = calc,
        trajectory=f"gcmc.traj",
        logfile=f"gcmc.log",
        loginterval=1,
        wrap_atoms = True
        )

dyn.run(10000)
