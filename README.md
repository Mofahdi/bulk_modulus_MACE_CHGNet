# bulk_modulus_MACE_CHGNet
code to calculate bulk modulus using machine learning potentials such as MACE and CHGNet</br>
methodology: the method depends on fitting several volumes with their total energies thorugh BirchMurnaghan fitting

## Usage
<code>
  python bulk_modulus_calc.py \
	--atoms_path='POSCAR' \
	--calculator='mace' \
	--dtype='float64' \
	--device='cpu' \
	--opt_atoms=True \
	--output_opt_atoms=True \
	--force_max=0.01 \
	--max_strain=0.1 \
	--num_samples=11 \
	--output_strains=False \
	--output_traj_strains=False \
</code>

## Args Explanation
**--atoms_path**: structure path ('./POSCAR' by default) \
**--calculator**: 'mace' or 'chgnet' are supported currently. Case does not matter ('mace' by default) \
**--dtype**: 'float64' or 'float32' for 'mace' calculator. If you select 'chgnet' calculator, then this command does not matter ('float64' by default) \
**--device**: 'cpu', 'mp', or 'cuda' for 'mace' calculator. If you select 'chgnet' calculator, then this command does not matter ('cpu' by default) \
**--opt_atoms**: whether you optimize the input structure. If True, the code will output the optimized structure in "POSCAR_opt" (True by default) \
**--output_opt_atoms**: if you optimize the input structure, you can choose whether you optimize the input structure (True by default) \
**--force_max**: (0.01 by default) \
**--max_strain**: (0.1 by default) \
**--num_samples**: (11 by default) \
**--output_strains**: (False by default) \
**--output_traj_strains**: (False by default) \
</br>
**Note:** you have to put both files "*mace_chgnet_classes.py*" and "*bulk_modulus_calc.py*" in the same path since "*bulk_modulus_calc.py*" inherits classes from "*mace_chgnet_classes.py*". The code will output the above files in the same path where you put "*mace_chgnet_classes.py*" and "*bulk_modulus_calc.py*".
</br>
**Another Note**: "*mace_chgnet_classes.py*" has other functions for optimization and md runs, but the main class inherited from it is "mace_EOS" starting in line 600 and ending in line 746 for the equation of state. You might observe lots of classes are for mace, and that is because the code was originally designed for mace but now it adopts chgnet as well for comparison purposes.

## Required Packages
the code is tested on the following packages and versions:
<code>torch=2.0.1</code>
<code>ase=3.23.0</code>
<code>pymatgen=2023.11.12</code>
<code>e3nn=0.4.4</code>
<code>mace-torch=0.3.6</code>
<code>chgnet=0.3.5</code>
<code>jarvis-tools=2024.4.30</code>
</br>The code can probably work with different versions of the above packages

## Credit
* Please consider reading my published work in Google Scholar using this [link](https://scholar.google.com/citations?user=5tkWy4AAAAAJ&hl=en&oi=ao) thank you :)
* also please let me know if more features are needed to be added and/or improved 
