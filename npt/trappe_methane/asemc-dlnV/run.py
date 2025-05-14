import numpy as np

from ase import Atoms, units
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib

from ase.mc.moveset import Moveset
from ase.mc.utility import random_packing
from ase.mc.moves import Volume, HMC
from ase.mc.mc import MonteCarlo

probe = Atoms("C", [[0., 0., 0.]])
atoms = Atoms(cell = [40.0, 40.0, 40.0], pbc=True)

n = 64
temperature_K = 150.0

sigma = 3.73
epsilon = 148.0*units.kB
rc = 14.0

#cmds = [ f"pair_style lj/cut {rc}",
#         f"pair_coeff * * {epsilon} {sigma}"]

#calc = LAMMPSlib(lmpcmds=cmds, log_file='lmp.log')

parameters = {
        'pair_style': f"lj/cut {rc}",
        'pair_coeff': [f"* * {epsilon} {sigma}"],
        'pair_modify': ["shift yes"],
        }


calc = LAMMPS(files=None, keep_alive = True, **parameters)

random_packing(atoms, probe, n, tolerance=2.0, max_iter=1000)

volume_move = Volume(
                accepted=0,
                attempted=0,
                probability=0.1,
                update_frequency=100,
                max_delta=0.01,
                beta=1.0 / units.kB / temperature_K,
                ln_volume=True,
                pressure=10.0 * units.bar,
                mask=None,
                scale_delta=0.1,
                target_acceptance=0.5,
            )

multiparticle_translate_move = HMC(
                accepted=0,
                attempted=0,
                probability=0.9,
                dt=5.0 * units.fs,
                nsteps=1,
                beta=1.0 / units.kB / temperature_K,
                r_overlap=0.5,
                update_frequency=1000,
                scale_delta=0.1,
                target_acceptance=0.7,
            )

moveset_npt = Moveset([volume_move, multiparticle_translate_move])
moveset_nvt = Moveset([multiparticle_translate_move])

dyn = MonteCarlo(
    atoms,
    moveset_npt,
    dft_calc = calc,
    trajectory="npt.traj",
    logfile="npt.log",
    loginterval=1,
    wrap_atoms=True,
    )

dyn.run(25000)


dyn = MonteCarlo(
    atoms,
    moveset_nvt,
    dft_calc = calc,
    trajectory="nvt.traj",
    logfile="nvt.log",
    loginterval=1,
    wrap_atoms=True
    )

dyn.run(75000)
