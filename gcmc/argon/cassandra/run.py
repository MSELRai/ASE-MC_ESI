import sys
import mbuild
import foyer
import mosdef_cassandra as mc
import unyt as u

import numpy as np
from scipy.constants import hbar, Avogadro, Boltzmann


def thermal_debroglie(mass_amu, temperature_K):
    """Returns thermal Debroglie wavelength in angstrom"""
    kg_to_g = 1000.0
    m_to_angstrom = 1e10

    mass_kg = mass_amu / Avogadro / kg_to_g
    debroglie_m = np.sqrt(
        2.0 * np.pi * hbar * hbar / mass_kg / Boltzmann / temperature_K
    )
    debroglie_angstrom = debroglie_m * m_to_angstrom
    return debroglie_angstrom


def adams_to_mu(adams, mass_amu, temperature_K, volume_angstrom3):
    tdw = thermal_debroglie(mass_amu, temperature_K)
    mu_over_kT = adams + 3.0*np.log(tdw) - np.log(volume_angstrom3)
    J_to_kJ = 1.0/1000.0
    mu_kJpermol = mu_over_kT * Boltzmann * temperature_K * J_to_kJ * Avogadro
    return mu_kJpermol


B = float(sys.argv[1])

temp = 120.0
boxl = 20.0
mu = adams_to_mu(B, 39.948, temp, boxl**3)

argon = mbuild.Compound(name="Ar")
ff = foyer.Forcefield("noble_gas.xml")
typed_argon = ff.apply(argon)

mols_to_add = [[int(np.exp(B))]]
species_list = [typed_argon]

moveset_gcmc = mc.MoveSet("gcmc", species_list)
moveset_gcmc.cbmc_n_insert = 10
moveset_gcmc.prob_translate = 0.5
moveset_gcmc.prob_insert = 0.5

custom_args = {
    "charge_style": "none",
    "rcut_min": 0.5 * u.angstrom,
    "vdw_cutoff": 10.0 * u.angstrom,
    "coord_freq": 1,
    "prop_freq": 1,
    "cutoff_style": "cut_shift",
    "chemical_potentials": [mu * (u.kJ / u.mol)]
}

vapor_box = mbuild.Box(lengths=[boxl/10.0, boxl/10.0, boxl/10.0])


box_list = [vapor_box]
system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

mc.run(
    run_name = "equil",
    system=system,
    moveset=moveset_gcmc,
    run_type="equilibration",
    run_length=100000,
    temperature= temp * u.K,
    **custom_args,
)
