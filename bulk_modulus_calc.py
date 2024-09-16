import argparse
import sys, os

from ase.io import read

from mace_chgnet_classes import *

# TODO: 1- add loading a mace or chgnet model feature if pretrained model is not preferred

def none_or_val(value):
	if value == 'None':
		return None
	return int(value)

def bool_vals(value):
	if value.lower() == 'true':
		return True
	if value.lower() == 'false':
		return False



parser = argparse.ArgumentParser(description='bulk modulus inputs')
parser.add_argument('--atoms_path', default='./POSCAR', type=str)
parser.add_argument('--calculator', default='mace', type=str)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--device', default='cpu', type=str)
#parser.add_argument('--model_path', default='./', type=str)
#parser.add_argument('--pretrained_model', default=True, type=bool_vals)
parser.add_argument('--opt_atoms', default=True, type=bool_vals)
parser.add_argument('--output_opt_atoms', default=True, type=bool_vals)
parser.add_argument('--force_max', default=0.01, type=float)

parser.add_argument('--max_strain', default=0.1, type=float)
parser.add_argument('--num_samples', default=11, type=int)
parser.add_argument('--output_strains', default=False, type=bool_vals)
parser.add_argument('--output_traj_strains', default=False, type=bool_vals)


args = parser.parse_args(sys.argv[1:])

atoms=read(args.atoms_path)

if args.calculator.lower()=='mace':
	calc=mace_mp(model="large", dispersion=False, default_dtype='float64', device='cpu')
elif args.calculator.lower()=='chgnet':
	calc=CHGNetCalculator()
else:
	raise ValueError("you can either select 'mace' or 'chgnet' with ignored case. In future version, we will add other MLP calculators")

base_EOS=mace_EOS(
		atoms, 
		convert_to_primitive=True, 
		calculator=calc, 
		optimize_input_atoms=True, 
		device=args.device, 
		default_dtype=args.dtype, 
		optimizer=BFGSLineSearch, 
		filter="FrechetCellFilter", 
		output_relaxed_structure=args.output_opt_atoms, # 'POSCAR_opt'
		trajectory=None, 
		logfile=None, 
		fmax=0.01, # force threshold in the relaxed cell
		)

base_EOS.fit(
		max_strain_val = args.max_strain, 
		num_samples=args.num_samples, 
		optimize_strained_atoms=True, 
		output_trajectory=False, 
		logfile = 'opt_strained.log', 
		output_optimized_strained_atoms=args.output_strains,
		)

print("bulk modulus =", base_EOS.get_bulk_modulus(unit="GPa"))
