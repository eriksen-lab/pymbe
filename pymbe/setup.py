#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module containing all initialization functions
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import parallel
import system
import calculation
import expansion
import kernel
import restart


def main():
		"""
		this function initializes and broadcasts mpi, mol, calc, and exp objects

		:return: mpi, mol, calc, and exp objects
		"""
		# mpi object
		mpi = parallel.MPICls()

		# mol object
		mol = _mol(mpi)

		# calc object
		calc = _calc(mpi, mol)

		# exp object
		mol, calc, exp = _exp(mpi, mol, calc)

		# bcast restart info
		exp = parallel.exp(mpi, calc, exp)

		return mpi, mol, calc, exp


def _mol(mpi):
		"""
		this function initializes mol object

		:param mpi: pymbe mpi object
		:return: pymbe mol object
		"""
		# mol object
		mol = system.MolCls()

		# input handling
		if mpi.master:

			# read input
			mol = system.set_system(mol)

			# translate input
			mol = system.translate_system(mol)

			# sanity check
			system.sanity_chk(mol)

		# bcast info from master to slaves
		mol = parallel.mol(mpi, mol)

		# make pyscf mol object
		mol.make(mpi)

		return mol


def _calc(mpi, mol):
		"""
		this function initializes calc object

		:param mpi: pymbe mpi object
		:param mol: pymbe mol object
		:return: pymbe calc object
		"""
		# calc object
		calc = calculation.CalcCls(mol)

		# input handling
		if mpi.master:

			# read input
			calc = calculation.set_calc(calc)

			# sanity check
			calculation.sanity_chk(mol, calc)

			# set target
			calc.target = [x for x in calc.target.keys() if calc.target[x]][0]

			# restart logical
			calc.restart = restart.restart()

			# put calc.mpi info into mpi object
			mpi.task_size = calc.mpi['task_size']

		# bcast info from master to slaves
		calc = parallel.calc(mpi, calc)

		return calc


def _exp(mpi, mol, calc):
		"""
		this function initializes exp object

		:param mpi: pymbe mpi object
		:param mol: pymbe mol object
		:param calc: pymbe calc object
		:return: updated mol and calc object and new pymbe exp object
		"""
		if mpi.master:

			# get ao integrals
			mol.hcore, mol.eri = kernel.ao_ints(mol)

			if calc.restart:

				# read fundamental info
				mol, calc = restart.read_fund(mol, calc)

				# get mo integrals
				mol.hcore, mol.vhf, mol.eri = kernel.mo_ints(mol, calc.mo_coeff)

				# exp object
				exp = expansion.ExpCls(mol, calc)

			else:

				# hf calculation
				mol.nocc, mol.nvirt, mol.norb, calc.hf, mol.e_nuc, \
					calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
					calc.occup, calc.orbsym, calc.mo_energy, calc.mo_coeff = kernel.hf(mol, calc)

				# reference and expansion spaces and mo coefficients
				calc.mo_energy, calc.mo_coeff, \
					calc.nelec, calc.ref_space, calc.exp_space = kernel.ref_mo(mol, calc)

				# get mo integrals
				mol.hcore, mol.vhf, mol.eri = kernel.mo_ints(mol, calc.mo_coeff)

				# base energy
				calc.prop['base']['energy'] = kernel.base(mol, calc)

				# exp object
				exp = expansion.ExpCls(mol, calc)

				# reference space properties
				calc.prop['ref'][calc.target] = kernel.ref_prop(mol, calc, exp)

				# write fundamental info
				restart.write_fund(mol, calc)

		# get dipole integrals
		mol.dipole = kernel.dipole_ints(mol) if calc.target in ['dipole', 'trans'] else None

		# bcast fundamental info
		mol, calc = parallel.fund(mpi, mol, calc)

		# exp object on slaves
		if not mpi.master:
			exp = expansion.ExpCls(mol, calc)

		# init tuples and hashes
		if mpi.master:
			exp.hashes, exp.tuples = expansion.init_tup(mol, calc)
			exp.min_order = exp.tuples[0].shape[1]
		else:
			exp.hashes = expansion.init_tup(mol, calc)[0]

		# restart
		if mpi.master:
			exp.start_order = restart.main(calc, exp)

		return mol, calc, exp


