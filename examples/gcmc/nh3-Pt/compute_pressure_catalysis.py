import sys

import numpy as np

from ase import units

from ase.io import read, write, iread
from ase.io import Trajectory

from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList

start = 0
traj = Trajectory("../gcmc.traj")[start:]
nframes = len(traj)


dz = 2.5

top_indices = np.arange(80, 96, dtype = int)
bottom_indices = np.arange(0, 16, dtype = int)
atoms_per_molecule = 4 # TO DO, Compute a neighbor list and count connected components instead
temperature = 1000.0
prefix = "NH3"

nmolecules = np.zeros(nframes)
#temperature = np.zeros(nframes)
volume = np.zeros(nframes)

def igl(volume, N, T):
    # PV = N * kb * T
    # P = N * kb * T / V
    # Use ASE units (P (eV/ angstrom^3))
    return N * units.kB * T / volume

frame = 0
for atoms in traj:
    if frame % 100 == 0:
        print(f"Processing frame {frame + 1} / {nframes}")
    atoms.wrap()
    cell = atoms.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))

    rz = atoms.get_positions()[:,2]
    rzi = rz[bottom_indices]
    rzj = rz[top_indices]
    rzk = rz[[atom.index for atom in atoms if atom.symbol == "N"]]

    rzi_avg = np.mean(rzi)
    rzj_avg = np.mean(rzj)

    invicinity = (rzi_avg - dz < rzk) & (rzk < rzj_avg + dz)
    not_invicinity = np.logical_not(invicinity)

    total_volume = atoms.get_volume()
    catalyst_plus_interface_volume = area * ((rzj_avg + dz) - (rzi_avg - dz))

    #temperature[frame] = atoms.get_temperature()
    nmolecules[frame] = np.sum(not_invicinity) # TO DO: Replace with a count over connected components
    volume[frame] = total_volume - catalyst_plus_interface_volume

    frame += 1

pressure = igl(volume, nmolecules, temperature)
pressure_bar = pressure / units.bar

np.save(f'{prefix}.pressure_bar.npy', pressure_bar)
np.save(f'{prefix}.nmolecules.npy', nmolecules)
np.save(f'{prefix}.volume.npy', volume)
