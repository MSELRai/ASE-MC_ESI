import numpy as np
from ase import build
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.constraints import FixInternals

from ase.calculators.espresso import Espresso, EspressoProfile

# https://www.materialscloud.org/discover/sssp/table/efficiency
pseudos = {
    'H': 'H.pbe-rrkjus_psl.1.0.0.UPF',  # 60, 480
    'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF', # 45, 360
}

espresso_args = {
    "control":{
        "pseudo_dir":'/home/wwilson/Programs/qe/SSSP/1.3.0/PBE/efficiency',
        "outdir":'./out',
        'calculation':'scf',
    },
    "system":{
        "ecutwfc": 60,
        "ecutrho": 480,
        "input_dft": "pbe",
        "nspin": 1,
        'occupations':'smearing',
        'smearing':'mp',
        'degauss':0.02
    },
    "electrons":{
        "conv_thr": 1.0E-9,
        "electron_maxstep": 200,
        "mixing_beta": 0.7
    }
}

dft_args = {
    "tstress":True,
    "tprnfor":True,
    "nosym": True,
    "kpts":(1,1,1),
    "input_data": espresso_args,
    "pseudopotentials": pseudos
}

calc = Espresso(
    profile=EspressoProfile(argv=["mpirun", "-np", "32", "pw.x"]),
    **dft_args
)


atoms = build.molecule("biphenyl")
atoms.center(vacuum=5.0)
atoms.set_pbc(True)

atoms.calc = calc

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
    traj.write(atoms)
