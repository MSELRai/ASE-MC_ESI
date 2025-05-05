import numpy as np
from ase import build, units
from ase.mc.moves import Thermal, HMC
from ase.mc.moveset import Moveset
from ase.mc.mc import MonteCarlo

from mace.calculators import mace_mp


class Torsion(Thermal):
    def __init__(
        self,
        beta,
        torsion_indices,
        mask,
        accepted=0,
        attempted=0,
        probability=1.0,
        update_frequency=5000,
        max_delta=180.0,
        target_acceptance=0.5,
        scale_delta=0.1,
        r_overlap=0.5,
        name="Torsion",
        ):

        Thermal.__init__(
            self,
            name=name,
            accepted=accepted,
            attempted=attempted,
            probability=probability,
            update_frequency=update_frequency,
            max_delta=max_delta,
            beta=beta,
            target_acceptance=target_acceptance,
            scale_delta=scale_delta,
            r_overlap=r_overlap,
            )

        self.torsion_indices = torsion_indices
        self.mask = mask

    def generate_r_n(self, atoms, r_o):
        zeta = self.generate_uniform_random_number(n=1)
        dphi = (2.0*zeta -1.0)*self.max_delta
        atoms.rotate_dihedral(*self.torsion_indices, dphi, mask=self.mask, indices=None)
        r_n = atoms.get_positions()
        return r_n


temperature_K = 300.0
atoms = build.molecule("biphenyl")
atoms.center(vacuum=5.0)
atoms.set_pbc(True)

calc = mace_mp()

mask = np.zeros(len(atoms), dtype=int)
for i in range(11,22):
    mask[i] = 1

torsion_move = Torsion(
    beta = 1.0/temperature_K/units.kB,
    torsion_indices = [5, 0, 14, 15],
    mask = mask,
)

hmc_move = HMC(
                accepted=0,
                attempted=0,
                probability=1.0,
                dt= 1.5 * units.fs,
                nsteps=1,
                beta=1.0/temperature_K/units.kB,
                r_overlap=0.5,
                update_frequency=500,
                scale_delta=0.1,
                target_acceptance=0.7,
)

moveset = Moveset([torsion_move, hmc_move])

dyn = MonteCarlo(
    atoms,
    moveset,
    dft_calc = calc,
    trajectory="nvt.traj",
    logfile="nvt.log",
    loginterval=1,
)

dyn.run(10000)
