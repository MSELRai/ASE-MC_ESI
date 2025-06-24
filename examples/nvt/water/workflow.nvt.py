import sys
import numpy as np
from ase import Atoms, units
from ase.io import read

from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from mace.calculators import mace_mp

from ase.md import Langevin

from ase.mc.ensembles import NVT
from ase.mc.moveset import Moveset
from ase.mc.mc import MonteCarlo

x = angleHOH * np.pi / 180.0 / 2.0
pos = [
        [0, 0, 0],
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        [0, rOH * np.cos(x), -rOH * np.sin(x)]
        ]

probe = Atoms('OH2', positions=pos)

boxlength = ((18.01528 / 6.022140857e23) / (1.00 / 1e24))**(1 / 3.)

probe.set_cell((boxlength, boxlength, boxlength))
probe.center()
atoms = probe.repeat((4,4,4))
atoms.set_pbc(True)

atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                    for i in range(len(atoms)//3)
                                    for j in [0, 1, 2]])

atoms.calc = TIP3P(rc=4.5)

md = Langevin(
        atoms, 1.0 * units.fs, temperature_K = 315, friction=0.01/units.fs,
        trajectory = "tip3p.traj", logfile='tip3p.log'
        )

md.run(10000)

del atoms.calc
del atoms.constraints

calc = mace_mp()

integrator = sys.argv[1] # md or mc

if integrator == "md":
    atoms.calc = calc
    md = Langevin(atoms, 1.0 * units.fs, temperature_K = 315.0, friction=0.01/units.fs,
            trajectory = "langevin.traj", logfile='langevin.log'
            )
    md.run(100000)

elif integrator == "mc":
    ensemble_nvt= NVT(temperature_K=315.0)
    moveset_nvt = Moveset(ensemble_nvt.get_moves())
    moveset_nvt.adjust_parameter("HMC", "target_acceptance", 0.5)
    dyn = MonteCarlo(
            atoms, moveset_nvt, dft_calc = calc,
    trajectory="nvt_mc.traj",
    logfile="nvt_mc.log",
    loginterval=1,
    )

dyn.run(100000)
