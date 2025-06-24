import numpy as np
from ase import build, units
from ase.optimize import BFGS
from gpaw import GPAW, PW
from ase.parallel import world
from ase.io.trajectory import Trajectory
from ase.constraints import FixInternals

atoms = build.molecule("biphenyl")
atoms.center(vacuum=5.0)
atoms.set_pbc(True)

atoms.calc = GPAW(mode=PW(60*units.Ry), xc='PBE', hund = False, txt=f'gpaw.txt')

phis = np.arange(0,360,1)
dihedral_indices = [5, 0, 14, 15]

mask = np.zeros(len(atoms), dtype=int)
for i in range(11,22):
    mask[i] = 1

traj = Trajectory("dihedral_scan.traj", mode="w")
for phi in phis:
    atoms.set_dihedral(*dihedral_indices, phi, mask)
    dihedral = [atoms.get_dihedral(*dihedral_indices), dihedral_indices]
    c = FixInternals(dihedrals_deg=[dihedral])
    atoms.set_constraint(c)
    opt = BFGS(atoms, trajectory=f'optimizations/dihedral_{phi}.traj', logfile=f'optimizations/dihedral_{phi}.log')
    opt.run(fmax=0.05)
    if world.rank == 0:
        traj.write(atoms)
