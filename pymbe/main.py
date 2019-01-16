#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import numpy as np
try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
	from pyscf import lib, scf
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')

import parallel
import system
import calculation
import expansion
import kernel
import driver
import tools
import restart
import results


def main():
		""" main program """
		# general settings and sanity checks
		_setup()
		# mpi, mol, calc, and exp objects
		mpi, mol, calc, exp = _init()
		# branch
		if not mpi.global_master:
			if mpi.local_master:
				# proceed to local master driver
				driver.master(mpi, mol, calc, exp)
			else:
				# proceed to slave driver
				driver.slave(mpi, mol, calc, exp)
		else:
			# proceed to main driver
			driver.main(mpi, mol, calc, exp)
			# print/plot results
			results.main(mpi, mol, calc, exp)
			# finalize
			parallel.final(mpi)


def _init():
		""" init mpi, mol, calc, and exp objects """
		# mpi, mol, and calc objects
		mpi = parallel.MPICls()
		mol = _mol(mpi)
		calc = _calc(mpi, mol)
		# set max_mem
		mol.max_memory = calc.misc['mem']
		# configure mpi
		parallel.set_mpi(mpi, calc)
		# exp object
		exp = _exp(mpi, mol, calc)
		# bcast restart info
		if mpi.parallel and calc.restart: parallel.exp(mpi, calc, exp, mpi.global_comm)
		return mpi, mol, calc, exp


def _mol(mpi):
		""" init mol object """
		# mol object
		mol = system.MolCls(mpi)
		parallel.mol(mpi, mol)
		mol.make(mpi)
		return mol


def _calc(mpi, mol):
		""" init calc object """
		# calc object
		calc = calculation.CalcCls(mpi, mol)
		parallel.calc(mpi, calc)
		return calc


def _exp(mpi, mol, calc):
		""" init exp object """
		if mpi.global_master:
			# restart
			if calc.restart:
				# get ao integrals
				mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
				# read fundamental info
				restart.read_fund(mol, calc)
				# exp object
				if calc.model['type'] != 'comb':
					exp = expansion.ExpCls(mol, calc, calc.model['type'])
				else:
					# exp.typ = 'occ' for occ-virt and exp.typ = 'virt' for virt-occ combined expansions
					raise NotImplementedError('combined expansions not implemented')
			# no restart
			else:
				# get ao integrals
				mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
				# hf calculation
				calc.hf, calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
					calc.occup, calc.orbsym, calc.mo = kernel.hf(mol, calc)
				# reference and expansion spaces
				calc.ref_space, calc.exp_space, \
					calc.no_exp, calc.no_act, calc.ne_act = kernel.active(mol, calc)
				# exp object
				if calc.model['type'] != 'comb':
					exp = expansion.ExpCls(mol, calc, calc.model['type'])
				else:
					# exp.typ = 'occ' for occ-virt and exp.typ = 'virt' for virt-occ combined expansions
					raise NotImplementedError('combined expansions not implemented')
				# reference mo coefficients
				calc.mo = kernel.ref_mo(mol, calc, exp)
				# base energy
				base = kernel.base(mol, calc, exp)
				calc.base['energy'] = base['energy']
				# zeroth-order energy
				calc.zero = kernel.zero_energy(mol, calc, exp)
				# write fundamental info
				restart.write_fund(mol, calc)
		else:
			# get ao integrals
			mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
		# bcast fundamental info
		parallel.fund(mpi, mol, calc)
		# restart and exp object on slaves
		if mpi.global_master:
			exp.min_order = restart.main(calc, exp)
		else:
			# exp object
			if calc.model['type'] != 'comb':
				exp = expansion.ExpCls(mol, calc, calc.model['type'])
			else:
				# exp.typ = 'virt' for occ-virt and exp.typ = 'occ' for virt-occ combined expansions
				raise NotImplementedError('comb expansion not implemented')
		return exp


def _setup():
		""" set general settings and perform sanity checks"""
		# force OMP_NUM_THREADS = 1
		lib.num_threads(1)
		# mute scf checkpoint files
		scf.hf.MUTE_CHKFILE = True
		# PYTHONHASHSEED
		pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
		try:
			if pythonhashseed == -1:
				raise RuntimeError('\nenvironment variable PYTHONHASHSEED appears not to have been set - \n'
									'please set this to an arbitrary integer, e.g., export PYTHONHASHSEED=0\n')
		except Exception as err:
			sys.stderr.write('\nRuntimeError : {0:}\n\n'.format(err))
			raise


if __name__ == '__main__':
	main()


