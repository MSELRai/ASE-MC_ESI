import sys
from ase import units
from ase.io import read, write
from mace.calculators import mace_mp


atoms = read("tip3p.traj", index=-1)
del atoms.calc
del atoms.constraints

calc = mace_mp()

integrator = sys.argv[1] # md or mc

if integrator == "md":
    from ase.md.npt import NPT
    atoms.calc = calc
    dt = 1.0 * units.fs
    integrator = NPT(
            atoms,
            dt,
            temperature_K = 315.0,
            externalstress = 1.0 * units.bar,
            ttime = 100.0 * dt,
            pfactor = 0.1 * (1000.0 * dt)**2,
            trajectory = "npt.traj",
            logfile='npt.log',
            loginterval=4,
    )

    integrator.run(500000)

elif integrator == "mc":
    from ase.mc.ensembles import NPT
    from ase.mc.moveset import Moveset
    from ase.mc.mc import MonteCarlo

    ensemble_npt= NPT(pressure=1.0*units.bar, temperature_K=315.0)
    moveset_npt = Moveset(ensemble_npt.get_moves())

    moveset_npt.adjust_parameter("Volume", "max_delta", 0.02)
    moveset_npt.adjust_parameter("Volume", "probability", 0.5)
    moveset_npt.adjust_parameter("HMC", "probability", 0.5)
    moveset_npt.adjust_parameter("Translate", "probability", 0.0)
    moveset_npt.adjust_parameter("Rotate", "probability", 0.0)

    dyn = MonteCarlo(
        atoms, moveset_npt, dft_calc = calc,
        trajectory="npt_mc.traj",
        logfile="npt_mc.log",
        loginterval=4,
    )
    dyn.run(500000)

