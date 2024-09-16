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
