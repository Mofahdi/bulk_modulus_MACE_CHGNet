import numpy as np
import os
import inspect
from typing import Union, TYPE_CHECKING, Optional

# units utils
from ase import Atoms, units
#from ase.units import kJ
from ase.io import read, write
from ase.io import Trajectory

# calculator utils
from mace.calculators import MACECalculator, mace_mp
from chgnet.model.dynamics import CHGNetCalculator
import ase.calculators.calculator
from ase.calculators.calculator import Calculator

# filter utils
from ase.constraints import ExpCellFilter, StrainFilter
import ase.filters as filters
from ase.filters import Filter
from ase.eos import EquationOfState

# md utils and simulations
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.npt import NPT

# Optimizer class and optimizers
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS, QuasiNewton
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.gpmin.gpmin import GPMin
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.optimize.oldqn import GoodOldQuasiNewton

# equation of state utils
from pymatgen.analysis.eos import BirchMurnaghan, Murnaghan, Birch
from jarvis.core.atoms import Atoms as JarvisAtoms # ase_to_atoms, ase_converter


class mace_optimize_md_runs:
	def __init__(self, 
			atoms, 
			calculator: Union[Calculator, str] = 'mace',
			dyn=None, 
			timestep: Optional[float] = None, 
			print_format = None,
			logger=None, 
			logfile="mace_md.log", 
			communicator=None,
			device='cpu',
			default_dtype="float64",
			bulk_modulus: Optional[float]=None,
			):

		self.atoms=atoms
		self.bulk_modulus=bulk_modulus
		self.default_dtype=default_dtype
		self.device=device
		if isinstance(calculator, str):
			if calculator.lower() == 'mace':
				calc = mace_mp(model="large", dispersion=False, default_dtype=self.default_dtype, device=self.device)
			elif calculator.lower() == 'chgnet':
				calc=CHGNetCalculator()
			else:
				raise ValueError('you must choose either: 1- "mace" or 2- "chgnet". The code only supports two calculators in str format, but you can use any ASE Calculator subclass format')
		
		if isinstance(calculator, Calculator) and not isinstance(calculator, str):
			calc = calculator

		#calc=CHGNetCalculator()
		#atoms.set_calculator(calculator) # this line is equivalent to "atoms.calc=calc"
		self.atoms.calc=calc

		self.print_format = print_format
		
		self.timestep=timestep
		self.logger=logger
		self.logfile = logfile
		self.communicator=communicator
		self.dyn=dyn


#		if self.print_format is None:
		self.print_format = self.example_print
#		else:
#			self.print_format=None

		if self.timestep is None:
			self.timestep = 0.1
		self.timestep = self.timestep * units.fs

		if self.logger is None:
			self.logger = MDLogger(
						self.dyn,
						self.atoms,
						self.logfile,
						stress=False,
						peratom=False,
						header=True,
						mode="w",
						)

	def optimize_structure(
			self,
			optimizer: Union[str, Optimizer, None] = "BFGSLineSearch",
			trajectory: Union[str, None] = "opt.traj", # use None to not output trajectory
			logfile="opt.log",
			steps: int = 1000,
			fmax: Union[float, int] = 1,
			optimize_lattice: bool = True,
			filter: str = "FrechetCellFilter", 
			interval: int = 1,
			output_relaxed_structure: bool = True,
			relaxed_filename: str = 'CONTCAR',
			):
		"""Optimize structure."""
		available_ase_optimizers = {
					"BFGS": BFGS,
					"LBFGS": LBFGS,
					"LBFGSLineSearch": LBFGSLineSearch,
					"FIRE": FIRE,
					"MDMin": MDMin,
					"GPMin": GPMin,
					"SciPyFminCG": SciPyFminCG,
					"SciPyFminBFGS": SciPyFminBFGS,
					"BFGSLineSearch": BFGSLineSearch,
					}
#		print(type(optimizer))
		if not isinstance(optimizer, str) and issubclass(optimizer, Optimizer):
			optimizer: Optimizer = optimizer
#			print('optimzer has type', type(optimizer))
		elif isinstance(optimizer, str):
#			print('str optimizer', optimizer, type(optimizer))
			optimizer = available_ase_optimizers.get(optimizer, None)
#			print(optimizer, type(optimizer))
#		else:
		if optimizer is None:
			raise ValueError(
					f"Optimizer instance not found. Select from {list(available_ase_optimizers)}"
					)
#		if optimizer is None:
#			raise ValueError(f"please select an optimizer")

		valid_filter_names = [
					name
					for name, cls in inspect.getmembers(filters, inspect.isclass)
					if issubclass(cls, Filter)
					]

		if isinstance(filter, str):
			if filter in valid_filter_names:
				filter = getattr(filters, filter)
			else:
				raise ValueError(
					f"Invalid {filter=}, must be one of {valid_filter_names}. "
						)

		if optimize_lattice:
			if filter is None:
				raise ValueError('select an ASE filter')
			#self.atoms = ExpCellFilter(self.atoms)
			#self.atoms = FrechetCellFilter(self.atoms)
			# FrechetCellFilter is better (default in this method) for convergence check out: https://wiki.fysik.dtu.dk/ase/_modules/ase/filters.html
			self.atoms = filter(self.atoms)
		

		print("OPTIMIZATION")
		if logfile is not None:
			self.dyn = optimizer(
						self.atoms, trajectory=trajectory, logfile=logfile
						)
		else:
			self.dyn = optimizer(self.atoms)

		if interval is not None:
			self.dyn.attach(self.print_format, interval=interval)

		self.dyn.run(fmax=fmax, steps=steps)

		if output_relaxed_structure:
			self.atoms.write(relaxed_filename, format='vasp', direct='True')

		return (
			self.atoms,
			self.atoms.get_potential_energy(),
			self.atoms.get_forces(),
			)


	# run NVE Verlet simulation
	def run_nve_velocity_verlet(
				self,
				filename="ase_nve",
				interval: int = 1,
				steps: int = 1000,
				initial_temperature_K: Union[int, float, None] = None,
				output_trajectory: bool = True,
					):
		"""Run NVE."""
		print("NVE VELOCITY VERLET")
		
		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
							temperature_K=initial_temperature_K
								)
		self.dyn = VelocityVerlet(self.atoms, self.timestep)

		# Create monitors for logfile and a trajectory file
		# logfile = os.path.join(".", "%s.log" % filename)
		if output_trajectory:
			trajfile = os.path.join(".", "%s.traj" % filename)
			trajectory = Trajectory(trajfile, "w", self.atoms)
			self.dyn.attach(trajectory.write, interval=interval)

		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.run(steps)
		return self.atoms


	# run NVT Langevin simulation
	def run_nvt_langevin(
			self,
			filename="ase_nvt_langevin",
			interval: int = 1,
			temperature_K: Union[int, float] = 300,
			steps: int = 1000,
			friction=1e-4,
			initial_temperature_K: Union[int, float, None] = None,
			output_trajectory: bool = True,
				):
		"""Run NVT."""
		print("NVT LANGEVIN")
		
		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
							temperature_K=initial_temperature_K
								)
		self.dyn = Langevin(
				self.atoms,
				self.timestep,
				temperature_K=temperature_K,
				friction=friction,
				communicator=self.communicator,
					)

		# Create monitors for logfile and a trajectory file
		# logfile = os.path.join(".", "%s.log" % filename)
		if output_trajectory:
			trajfile = os.path.join(".", "%s.traj" % filename)
			trajectory = Trajectory(trajfile, "w", self.atoms)
			self.dyn.attach(trajectory.write, interval=interval)

		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.run(steps)
		return self.atoms


	# run NVT Andersen simulation
	def run_nvt_andersen(
			self,
			filename="ase_nvt_andersen",
			interval: int = 1,
			initial_temperature_K: Union[int, float, None] = None,
			temperature_K: Union[int, float] = 300,
			steps: int =1000,
			andersen_prob=1e-1,
			output_trajectory: bool = True,
				):
		"""Run NVT."""
		print("NVT ANDERSEN")
		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
							temperature_K=initial_temperature_K
								)
		self.dyn = Andersen(
				self.atoms,
				self.timestep,
				temperature_K=temperature_K,
				andersen_prob=andersen_prob,
				communicator=self.communicator,
					)

		# Create monitors for logfile and a trajectory file
		# logfile = os.path.join(".", "%s.log" % filename)
		if output_trajectory:
			trajfile = os.path.join(".", "%s.traj" % filename)
			trajectory = Trajectory(trajfile, "w", self.atoms)
			self.dyn.attach(trajectory.write, interval=interval)

		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.run(steps)
		return self.atoms


	def run_nvt_berendsen(
				self,
				filename="ase_nvt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=1000,
				taut=None,
				):
		"""Run NVT."""
		print("NVT BERENDSEN")
		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
								temperature_K=initial_temperature_K
								)
		if taut is None:
			taut = 100 * self.timestep
		self.dyn = NVTBerendsen(
					self.atoms,
					self.timestep,
					temperature_K=temperature_K,
					taut=taut,
					communicator=self.communicator,
					)
		# Create monitors for logfile and a trajectory file
		trajfile = os.path.join(".", "%s.traj" % filename)
		trajectory = Trajectory(trajfile, "w", self.atoms)
		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.attach(trajectory.write, interval=interval)
		self.dyn.run(steps)
		return self.atoms

	def run_npt_berendsen(
				self,
				filename="ase_npt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=1000,
				taut= None, #49.11347394232032,
				taup= None, #98.22694788464064,
				pressure=1.01325e-4,
				compressibility_au=None,
				):
		"""Run NPT."""
		print("NPT BERENDSEN")

		if hasattr(self, 'bulk_modulus') and self.bulk_modulus is not None:
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au
		else:
			print('bulk modulus does not exist. Therfore, it will be calculated now\n')
			base_EOS=mace_EOS(self.atoms, convert_to_primitive=True, calculator=self.atoms.calc, optimize_input_atoms=True, device=self.device, 
					default_dtype=self.default_dtype, optimizer='BFGSLineSearch', filter="FrechetCellFilter", 
					output_relaxed_structure=False, trajectory=None, logfile=None, fmax=0.1)

			base_EOS.fit(optimize_strained_atoms=True, output_trajectory=False, logfile = 'opt_strained.log', output_optimized_strained_atoms=False)
			self.bulk_modulus = float(base_EOS.get_bulk_modulus(unit="GPa"))
			print('bulk modulus is calculated to be: '+str(self.bulk_modulus)+' GPa\n\n')
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au

		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
								temperature_K=initial_temperature_K
							)

		if taut is None:
			taut = 100 * self.timestep
		if taup is None:
			taup = 1000 * self.timestep
#		print()
		self.dyn = NPTBerendsen(
					self.atoms,
					self.timestep,
					temperature_K=temperature_K,
					taut=taut,
					taup=taup,
					pressure_au=pressure * units.GPa,
					#pressure=pressure,
					compressibility_au=compressibility_au,
					)

		# Create monitors for logfile and a trajectory file
		trajfile = os.path.join(".", "%s.traj" % filename)
		trajectory = Trajectory(trajfile, "w", self.atoms)
		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.attach(trajectory.write, interval=interval)
		self.dyn.run(steps)
		return self.atoms


	def run_Inhomogeneous_npt_berendsen(
					self,
					filename="ase_Inhomogeneous_npt_berendsen",
					interval=1,
					initial_temperature_K=None,
					temperature_K=300,
					steps=1000,
					taut= None, #49.11347394232032,
					taup= None, #98.22694788464064,
					pressure=1.01325e-4,
					compressibility_au=None,
					):
		"""Run NPT."""
		print("NPT INHOMOGENEOUS BERENDSEN")

		if hasattr(self, 'bulk_modulus') and self.bulk_modulus is not None:
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au
		else:
			print('bulk modulus does not exist. Therfore, it will be calculated now\n')
			base_EOS=mace_EOS(self.atoms, convert_to_primitive=True, calculator=self.atoms.calc, optimize_input_atoms=True, device=self.device, 
					default_dtype=self.default_dtype, optimizer='BFGSLineSearch', filter="FrechetCellFilter", 
					output_relaxed_structure=False, trajectory=None, logfile=None, fmax=0.1)

			base_EOS.fit(optimize_strained_atoms=True, output_trajectory=False, logfile = 'opt_strained.log', output_optimized_strained_atoms=False)
			self.bulk_modulus = float(base_EOS.get_bulk_modulus(unit="GPa"))
			print('bulk modulus is calculated to be: '+str(self.bulk_modulus)+' GPa\n\n')
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au

		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
								temperature_K=initial_temperature_K
							)

		if taut is None:
			taut = 100 * self.timestep
		if taup is None:
			taup = 1000 * self.timestep
#		print()
		self.dyn = Inhomogeneous_NPTBerendsen(
						self.atoms,
						self.timestep,
						temperature_K=temperature_K,
						taut=taut,
						taup=taup,
						pressure_au=pressure * units.GPa,
						#pressure=pressure,
						compressibility_au=compressibility_au,
						)

		# Create monitors for logfile and a trajectory file
		trajfile = os.path.join(".", "%s.traj" % filename)
		trajectory = Trajectory(trajfile, "w", self.atoms)
		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.attach(trajectory.write, interval=interval)
		self.dyn.run(steps)
		return self.atoms


	def run_npt_nose_hoover(
				self,
				filename="ase_npt_nose_hoover",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=1000,
				pressure=1.01325e-4,
				taut=None,
				taup=None,
				):
		"""Run NPT."""
		print("NPT: Combined Nose-Hoover and Parrinello-Rahman dynamics")

		if hasattr(self, 'bulk_modulus') and self.bulk_modulus is not None:
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au
		else:
			print('bulk modulus does not exist. Therfore, it will be calculated now\n')
			base_EOS=mace_EOS(self.atoms, convert_to_primitive=True, calculator=self.atoms.calc, optimize_input_atoms=True, device=self.device, 
					default_dtype=self.default_dtype, optimizer='BFGSLineSearch', filter="FrechetCellFilter", 
					output_relaxed_structure=False, trajectory=None, logfile=None, fmax=0.1)

			base_EOS.fit(optimize_strained_atoms=True, output_trajectory=False, logfile = 'opt_strained.log', output_optimized_strained_atoms=False)
			self.bulk_modulus = float(base_EOS.get_bulk_modulus(unit="GPa"))
			print('bulk modulus is calculated to be: '+str(self.bulk_modulus)+' GPa\n\n')
			bulk_modulus_au = self.bulk_modulus / 160.2176  # GPa to eV/A^3
			compressibility_au = 1 / bulk_modulus_au


		if initial_temperature_K is not None:
			self.set_momentum_maxwell_boltzmann(
							temperature_K=initial_temperature_K
								)

		if taut is None:
			taut = 100 * self.timestep
		if taup is None:
			taup = 1000 * self.timestep
		ptime = taup
		self.upper_triangular_cell()
		self.dyn = NPT(
				self.atoms,
				self.timestep,
				temperature_K=temperature_K,
				externalstress=pressure * units.GPa,
				ttime=taut * units.fs,
				pfactor=self.bulk_modulus * units.GPa * ptime * ptime,
				)
		# Create monitors for logfile and a trajectory file
		trajfile = os.path.join(".", "%s.traj" % filename)
		trajectory = Trajectory(trajfile, "w", self.atoms)
		# Attach monitors to trajectory
		self.dyn.attach(self.logger, interval=interval)
		self.dyn.attach(self.print_format, interval=interval)
		self.dyn.attach(trajectory.write, interval=interval)
		self.dyn.run(steps)
		return self.atoms

	def upper_triangular_cell(self, *, verbose: Union[bool, None] = False) -> None:
		"""Transform to upper-triangular cell.
		ASE Nose-Hoover implementation only supports upper-triangular cell
		while ASE's canonical description is lower-triangular cell.

		Args:
			verbose (bool): Whether to notify user about upper-triangular cell
				transformation. Default = False
		"""
		if not NPT._isuppertriangular(self.atoms.get_cell()):  # noqa: SLF001
			a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
			angles = np.radians((alpha, beta, gamma))
			sin_a, sin_b, _sin_g = np.sin(angles)
			cos_a, cos_b, cos_g = np.cos(angles)
			cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
			cos_p = np.clip(cos_p, -1, 1)
			sin_p = (1 - cos_p**2) ** 0.5

			new_basis = [
				(a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
				(0, b * sin_a, b * cos_a),
				(0, 0, c),
				]

			self.atoms.set_cell(new_basis, scale_atoms=True)
			if verbose:
				print("Transformed to upper triangular unit cell.", flush=True)


	def example_print(self):
		"""Print info."""
		#if isinstance(self.atoms, ExpCellFilter):
		#if isinstance(self.atoms, FrechetCellFilter):
		if issubclass(type(self.atoms), Filter):
			self.atoms = self.atoms.atoms
		line = ""
		try:
			line = f"time={self.dyn.get_time() / units.fs: 5.0f} fs "
		except Exception:
			pass
		line += (
			f"a={self.atoms.cell.cellpar()[0]: 3.4f} Ang "
			+ f"b={self.atoms.cell.cellpar()[1]: 3.4f} Ang "
			+ f"c={self.atoms.cell.cellpar()[2]: 3.4f} Ang "
			+ f"Volume={self.atoms.get_volume(): 5.6f} amu/a3 "
			+ f"PE={self.atoms.get_potential_energy(): 5.6f} eV "
			+ f"KE={self.atoms.get_kinetic_energy(): 5.6f} eV "
			+ f"T={self.atoms.get_temperature(): 3.3f} K "
			# + f" P={atoms.
			# get_isotropic_pressure(atoms.get_stress()): 5.3f} bar "
			)
		print(line)

	def set_momentum_maxwell_boltzmann(
					self, 
					temperature_K: Union[int, float, None] = 10,
						):
		"""Set initial temperature."""
		print("SETTING INITIAL TEMPERATURE K", temperature_K)
		MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)


def ase_to_jarvis(ase_atoms):
	"""Convert ASE Atoms to JARVIS."""
	return JarvisAtoms(
			lattice_mat=ase_atoms.get_cell(),
			elements=ase_atoms.get_chemical_symbols(),
			coords=ase_atoms.get_positions(),
			#         pbc=True,
			cartesian=True,
			)

class mace_EOS:
	def __init__(self, 
		atoms: Atoms,
		convert_to_primitive: bool = True,
		calculator: Union[Calculator, str] = 'mace',
		optimize_input_atoms: bool = True,
		device: str ='cpu',
		default_dtype: str = 'float64', 
		optimizer: Union[str, Optimizer, None] = 'BFGSLineSearch',
		filter: str = "FrechetCellFilter",
		output_relaxed_structure=True,
		trajectory=None,
		logfile=None,
		fmax=0.1, 
			):

		if convert_to_primitive:
			j_atoms=ase_to_jarvis(atoms)
			j_atoms=j_atoms.get_primitive_atoms
			atoms = j_atoms.ase_converter()

		self.optimizer = optimizer
		self.device=device
		self.default_dtype=default_dtype
		self.fmax=fmax

		if isinstance(calculator, str):
			if calculator.lower() == 'mace':
				self.calc = mace_mp(model="large", dispersion=False, default_dtype=default_dtype, device=device)
			elif calculator.lower() == 'chgnet':
				self.calc=CHGNetCalculator()
			else:
				raise ValueError('you must choose either: 1- "mace" or 2- "chgnet". The code only supports two calculators in str format, but you can use any ASE Calculator subclass format')
		
		if isinstance(calculator, Calculator) and not isinstance(calculator, str):
			self.calc = calculator

		if optimize_input_atoms:
			self.atoms_optimized = self.mace_optimize(atoms, filter=filter, output_relaxed_structure=output_relaxed_structure)
		else:
			self.atoms_optimized=atoms
			self.atoms_optimized.calc=self.calc

		self.fitted = False
	
	def mace_optimize(self, 
			atoms: Atoms, 
			fmax=0.1, 
			trajectory="opt.traj", 
			logfile='opt.log', 
			steps=1000, 
			optimize_lattice: bool = True, 
			filter="FrechetCellFilter", 
			interval: int = 1, 
			output_relaxed_structure: bool = True,
			relaxed_filename: str = 'POSCAR_opt',
			):

		optimize_run_setup = mace_optimize_md_runs(atoms, 
							calculator=self.calc, 
							logfile=None,
							device=self.device, # you can use 'cuda' if you want to run on GPU
							default_dtype=self.default_dtype, # you can use "float32" for faster but less accurate results (not recommended for optimization)
							)

		atoms_optimized, PE_optimized, forces_optimized=optimize_run_setup.optimize_structure(optimizer = self.optimizer, 
													trajectory=trajectory, 
													logfile=logfile, 
													steps=steps, 
													fmax=self.fmax,
													optimize_lattice=optimize_lattice, 
													filter = "FrechetCellFilter",
													interval=1,
													output_relaxed_structure=output_relaxed_structure,
													relaxed_filename='POSCAR_opt',
													)

		
		return atoms_optimized
		
	def apply_strain(self, 
			strain: Union[int, float, list], 
			inplace: bool = False
			):
		strain_matrix = (1 + np.array(strain)) * np.eye(3)
		new_cell= np.dot(self.atoms_optimized.get_cell().T, strain_matrix).T
		strained_atoms=self.atoms_optimized if inplace else self.atoms_optimized.copy()
		strained_atoms.cell=new_cell
		return strained_atoms

	def fit(self,  
		max_strain_val = 0.1,
		num_samples: int = 11, 
		output_trajectory: bool = False, 
		logfile = None, 
		fmax=0.1, 
		optimize_lattice=False, 
		optimize_strained_atoms: bool = True, 
		output_optimized_strained_atoms: bool = False,
		):	
		volumes, energies = [], []
		if  optimize_lattice:
			raise ValueError('you cannot optimize lattice to get bulk modulus and compressibility')
		for strain_value in np.linspace(-max_strain_val, max_strain_val, num_samples):
			strained_atoms = self.apply_strain(strain = [strain_value, strain_value, strain_value], inplace=False)
			if optimize_strained_atoms:
				print('\noptimize strain:', strain_value)
				if logfile is not None and output_trajectory:
					strained_atoms_opt=self.mace_optimize(strained_atoms, trajectory='opt_'+str(strain_value)+'.traj', logfile=logfile, 
										optimize_lattice=optimize_lattice, output_relaxed_structure=False,)
				elif logfile is not None and not output_trajectory:
					strained_atoms_opt=self.mace_optimize(strained_atoms, trajectory=None, logfile=logfile, 
										optimize_lattice=optimize_lattice, output_relaxed_structure=False,)
			else:
				strained_atoms_opt=strained_atoms.copy()
				strained_atoms_opt.calc=self.calc

			if output_optimized_strained_atoms:
				strained_atoms_opt.write('POSCAR_strained_'+str(strain_value), format='vasp', direct='True')

			volumes.append(strained_atoms_opt.get_volume())
			energies.append(strained_atoms_opt.get_total_energy())


		self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
		self.bm.fit()
		self.fitted = True

	def get_bulk_modulus(self, unit: str = "eV/A^3") -> float:
		if self.fitted is False:
			raise ValueError('Equation of state needs to be fitted first through the following line: "self.fit(...)"')

		if unit == "eV/A^3":
			return self.bm.b0
		if unit == "GPa":
			return self.bm.b0_GPa

	def get_compressibility(self, unit: str = "A^3/eV") -> float:
		if self.fitted is False:
			raise ValueError('Equation of state needs to be fitted first through the following line: "self.fit(...)"')
		if unit == "A^3/eV":
			return 1 / self.bm.b0
		if unit == "GPa^-1":
			return 1 / self.bm.b0_GPa
		if unit in {"Pa^-1", "m^2/N"}:
			return 1 / (self.bm.b0_GPa * 1e9)
		raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")

#atoms=read('Si_conv_POSCAR_opt')
#mace_MP=mace_mp(model="large", dispersion=False, default_dtype='float64', device='cpu')
#mace_MP=CHGNetCalculator()

#base_EOS=mace_EOS(atoms, convert_to_primitive=True, calculator=mace_MP, optimize_input_atoms=True, device='cpu', default_dtype='float64', 
#		optimizer=BFGSLineSearch, filter="FrechetCellFilter", output_relaxed_structure=False, trajectory=None, logfile=None, fmax=0.1)
#base_EOS.fit(max_strain_val= 0.1, num_samples=11, optimize_strained_atoms=True, output_trajectory=False, logfile = 'opt_strained.log', output_optimized_strained_atoms=False)
#print(base_EOS.get_bulk_modulus(unit="GPa"))
#exit()
#if issubclass(BFGSLineSearch, Optimizer):
#	print(True)
#print(BFGSLineSearch)
#print()'BFGSLineSearch' GoodOldQuasiNewton LBFGSLineSearch
#print(mace_EOS(atoms, mace_mp, device='cpu', default_dtype='float64', optimizer=QuasiNewton).atoms_optimized)

"""
	def optimize_structure(
			self,
			optimizer: Union[str, Optimizer, None] = "BFGS",
			trajectory: Union[str, None] = "opt.traj", # use None to not output trajectory
			logfile="opt.log",
			steps: int = 1000,
			fmax: Union[float, int] = 1,
			optimize_lattice: bool = True,
			filter: str = "FrechetCellFilter", 
			interval: int = 1,
			output_relaxed_structure: bool = True,
			relaxed_filename: str = 'CONTCAR',
			):

"""

#exit()			

if __name__=='__main__':


#	calc = mace_mp(model="large", dispersion=False, default_dtype=default_dtype, device=device)
#	EOS(calculator = calc, device='cpu', default_dtype='float64', optimizer='BFGS')
	




	#exit()
	# the steps to run the code are as follows:
	# 1- get structure format in ASE format
	# 2- class initialization
	# Note: The above two steps are a "must" at all times. However, the steps below are optional in no particular order
	# 3- optimization
	# 4- md NVE 
	# 5- md NVT Langevin
	# 6- md NVT Andersen
	# 7- md NVT Berendsen
	# 8- md NPT Berendsen
	# 9- md inhomogeneous NPT Berendsen
	# 10- md NPT combined Noose-Hoover and Parrinello-Rahman dynamics with upper-triangular cell

	# 1- get structure format in ASE format
	# Note: pymatgen and jarvis structure format do not work
	atoms=read('Si_pri_POSCAR_95')

	# 2- class initialization
	md_run=mace_optimize_md_runs(atoms, 
					calculator='mace',
					dyn=None, 
					timestep=1, # units are forced to be in fs. default timestep=0.01 if left as "None", logger
					logger=None, 
					print_format = False, # "False" means you dont want to print anything, set it to "None" if you want to print something
					logfile="mace_md.log", # you can set it to None if you just want to optimize structures
					device='cpu', # you can use 'cuda' if you want to run on GPU
					default_dtype="float64", # you can use "float32" for faster but less accurate results
					bulk_modulus=95, # default bulk modulus is "None" (input must be in GPa). If you run NPT and bulk modulus is None then it will be calculated
					) 


	# Note: initialize another class mace_optimizer_md_runs if you dont want the same run
	# for example if you want to run NVT Langevin separately from NVT Andersen then you should initialize the class "mace_optimizer_md_runs" again
	# with a different variable name from the above "md_run", 
	# so the class variables with their runs will become something like "md_run1.run_nvt_langevin(...)" and "md_run2.run_nvt_andersen(...)"
	# where "md_run1" and "md_run2" are different class initializations

	# Mind: if you want to optimize then run NVE, NVT, NPT, or all, YOU MUST USE THE SAME CLASS VARIABLE


	# 3- optimization
	atoms, PE, forces=md_run.optimize_structure(optimizer="LBFGS", 
							trajectory="opt.traj", # use None to not output trajectory  
							logfile="opt.log", 
							steps=100, 
							fmax=0.1, 
							optimize_lattice=True, 
							filter = "FrechetCellFilter", # you can also use "UnitCellFilter"  
							interval=1,
							output_relaxed_structure=False,
							relaxed_filename='POSCAR_opt',
							)


	"""
	# 7- NVT Berendsen
	md_run.run_nvt_berendsen(
				filename="ase_nvt_berendsen",
				interval=1,
				initial_temperature_K=1,
				temperature_K=300,
				steps=10,
				taut=None,
				)

	# 8- NPT Berendsen
	md_run.run_npt_berendsen(
				filename="ase_npt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				taut= None, #49.11347394232032,
				taup= None, #98.22694788464064,
				pressure=1.01325e-4,
				#pressure=1.01325, 
				compressibility_au=None,
				)

	# 9- inhomogeneous NPT Berendsen	
	md_run.run_Inhomogeneous_npt_berendsen(
				filename="ase_Inhomogeneous_npt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				taut= None, #49.11347394232032,
				taup= None, #98.22694788464064,
				pressure=1.01325e-4,
				#pressure=1.01325, 
				compressibility_au=None,
				)

	# 10- NPT combined Noose-Hoover and Parrinello-Rahman dynamics with upper-triangular cell
	md_run.run_npt_nose_hoover(
				filename="ase_npt_nose_hoover",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				pressure=1.01325e-4,
				taut=None,
				taup=None,
				)


	exit()
	"""

	"""
	available_ase_optimizers = {
					"BFGS": BFGS,
					"LBFGS": LBFGS,
					"LBFGSLineSearch": LBFGSLineSearch,
					"FIRE": FIRE,
					"MDMin": MDMin,
					"GPMin": GPMin,
					"SciPyFminCG": SciPyFminCG,
					"SciPyFminBFGS": SciPyFminBFGS,
					"BFGSLineSearch": BFGSLineSearch,
					}

	"""

#	exit()	
	# 4- md NVE
#	md_run.run_nve_velocity_verlet(filename="ase_nve", 
#					interval=1, 
#					steps=10.1, 
#					initial_temperature_K=5,
#					output_trajectory = True,
#					)

	md_run.run_nve_velocity_verlet(filename="ase_nve", 
					interval=1, 
					steps=5, 
					initial_temperature_K=5,
					output_trajectory = True,
					)

#	exit()
	# 5- md NVT Langevin
	md_run.run_nvt_langevin(filename="ase_nvt_langevin", 
				interval=1, 
				temperature_K=100, 
				steps=10, 
				friction=1e-4, 
				initial_temperature_K=5,
				output_trajectory = True,
				)

	# 6- md NVT Andersen 
	md_run.run_nvt_andersen(filename="ase_nvt_andersen", 
				interval=1, 
				temperature_K=100, 
				steps=10, 
				andersen_prob=1e-1, 
				initial_temperature_K=None,
				output_trajectory = True,
				)


	# 7- NVT Berendsen
	md_run.run_nvt_berendsen(
				filename="ase_nvt_berendsen",
				interval=1,
				initial_temperature_K=1,
				temperature_K=300,
				steps=10,
				taut=None,
				)


	# 8- NPT Berendsen
	md_run.run_npt_berendsen(
				filename="ase_npt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				taut= None, #49.11347394232032,
				taup= None, #98.22694788464064,
				pressure=1.01325e-4,
				#pressure=1.01325, 
				compressibility_au=None,
				)

	# 9- inhomogeneous NPT Berendsen	
	md_run.run_Inhomogeneous_npt_berendsen(
				filename="ase_Inhomogeneous_npt_berendsen",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				taut= None, #49.11347394232032,
				taup= None, #98.22694788464064,
				pressure=1.01325e-4,
				#pressure=1.01325, 
				compressibility_au=None,
				)

	# 10- NPT combined Noose-Hoover and Parrinello-Rahman dynamics with upper-triangular cell
	md_run.run_npt_nose_hoover(
				filename="ase_npt_nose_hoover",
				interval=1,
				initial_temperature_K=None,
				temperature_K=300,
				steps=10,
				pressure=1.01325e-4,
				taut=None,
				taup=None,
				)

