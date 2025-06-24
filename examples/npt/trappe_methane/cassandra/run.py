import mbuild
import foyer
import mosdef_cassandra as mc
import unyt as u

# Use mbuild to create a coarse-grained CH4 bead
methane = mbuild.Compound(name="_CH4")
trappe = foyer.forcefields.load_TRAPPE_UA()
typed_methane = trappe.apply(methane)

mols_to_add = [[64]]
species_list = [typed_methane]

moveset_npt = mc.MoveSet("npt", species_list)
moveset_nvt = mc.MoveSet("nvt", species_list)



custom_args = {
    "charge_style": "none",
    "rcut_min": 2.0 * u.angstrom,
    "vdw_cutoff": 14.0 * u.angstrom,
    "units": "sweeps",
    "steps_per_sweep": 64,
    "coord_freq": 1,
    "prop_freq": 1,
    "cutoff_style": "cut_shift",
    }

vapor_box = mbuild.Box(lengths=[4.0, 4.0, 4.0])
box_list = [vapor_box]
system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

mc.run(
    run_name = "equil",
    system=system,
    moveset=moveset_npt,
    run_type="equilibration",
    run_length=25000,
    pressure=10.0 * u.bar,
    temperature=150.0 * u.K,
    **custom_args,
)



import pandas as pd
def read_cassandra_prp(filename):
    with open(filename) as fh:
        #First line
        fh.readline()
        #Secondline
        column_names = fh.readline().split()
        del column_names[0]
    return pd.read_table(filename, skiprows=3, names=column_names, delim_whitespace=True)

df_eq = read_cassandra_prp("equil.out.prp")


average_volume = df_eq[df_eq['MC_SWEEP'] >= 12500]["Volume"].mean()
average_boxl = average_volume**(1.0/3.0) / 10.0 # Cassandra reports angstrom but we require nm in MoSDeF

vapor_box = mbuild.Box(lengths=[average_boxl, average_boxl, average_boxl])
box_list = [vapor_box]
system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

mc.run(
    run_name = "prod",
    system=system,
    moveset=moveset_nvt,
    run_type="equilibration",
    run_length=75000,
    temperature=150.0 * u.K,
    **custom_args,
)


