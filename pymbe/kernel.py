#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import shutil
import copy
import numpy as np
import scipy as sp
from functools import reduce
from mpi4py import MPI
from pyscf import gto, symm, scf, ao2mo, lo, ci, cc, mcscf, fci

import tools


SPIN_TOL = 1.0e-05


def ao_ints(mol, calc):
		""" get AO integrals """
		# core hamiltonian and electron repulsion ints
		if mol.atom: # ab initio hamiltonian
			hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
			if mol.cart:
				eri = mol.intor('int2e_cart', aosym=4)
			else:
				eri = mol.intor('int2e_sph', aosym=4)
		else: # model hamiltonian
			hcore = _hubbard_h1(mol)
			eri = _hubbard_eri(mol)
		# dipole integrals with gauge origin at origo
		if calc.target['dipole'] or calc.target['trans']:
			with mol.with_common_origin([0.0, 0.0, 0.0]):
				dipole = mol.intor_symmetric('int1e_r', comp=3)
		else:
			dipole = None
		return hcore, eri, dipole


def _hubbard_h1(mol):
		""" set hubbard hopping hamiltonian """
		# 1d
		if mol.dim == 1:
			# init
			n = mol.nsites
			t = mol.t
			h1 = np.zeros([n]*2, dtype=np.float64)
			# adjacent neighbours
			for i in range(n-1):
				h1[i, i+1] = h1[i+1, i] = -t
			# pbc
			if mol.pbc:
				h1[-1, 0] = h1[0, -1] = -t
		# 2d
		elif mol.dim == 2:
			# init
			n = int(np.sqrt(mol.nsites))
			t = mol.t
			h1 = np.zeros([n**2]*2, dtype=np.float64)
			# adjacent neighbours - sideways
			for i in range(n**2):
				if i % n == 0:
					h1[i, i+1] = -t
				elif i % n == n-1:
					h1[i, i-1] = -t
				else:
					h1[i, i-1] = h1[i, i+1] = -t
			# adjacent neighbours - up-down
			for i in range(n**2):
				if i < n:
					h1[i, i+n] = -t
				elif i >= n**2 - n:
					h1[i, i-n] = -t
				else:
					h1[i, i-n] = h1[i, i+n] = -t
			# pbc
			if mol.pbc:
				# sideways
				for i in range(n):
					h1[i*n, i*n+(n-1)] = h1[i*n+(n-1), i*n] = -t
				# up-down
				for i in range(n):
					h1[i, n*(n-1)+i] = h1[n*(n-1)+i, i] = -t
		return h1


def _hubbard_eri(mol):
		""" set hubbard two-electron hamiltonian """
		# init
		n = mol.nsites
		u = mol.u
		eri = np.zeros([n]*4, dtype=np.float64)
		for i in range(n):
			eri[i,i,i,i] = u
		return eri


def hf(mol, calc):
		""" hartree-fock calculation """
		# perform restricted hf calc
		hf = scf.RHF(mol)
		hf.conv_tol = 1.0e-09
		hf.max_cycle = 500
		if mol.atom: # ab initio hamiltonian
			hf.irrep_nelec = mol.irrep_nelec
		else: # model hamiltonian
			hf.get_ovlp = lambda *args: np.eye(mol.nsites)
			hf.get_hcore = lambda *args: mol.hcore 
			hf._eri = mol.eri
		# perform hf calc
		for i in list(range(0, 12, 2)):
			hf.diis_start_cycle = i
			try:
				hf.kernel()
			except sp.linalg.LinAlgError: pass
			if hf.converged: break
		# convergence check
		if not hf.converged:
			try:
				raise RuntimeError('\nHF Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# dipole moment
		if calc.target['dipole']:
			dm = hf.make_rdm1()
			elec_dipole = np.einsum('xij,ji->x', mol.dipole, dm)
			elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		else:
			elec_dipole = None
		# determine dimensions
		mol.norb, mol.nocc, mol.nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		if mol.atom:
			orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
			wfnsym = 0
			for ir in orbsym[occup == 1]:
				wfnsym ^= ir
			# sanity check
			if wfnsym != calc.state['wfnsym'] and calc.no_exp == 0:
				try:
					raise RuntimeError('\nHF Error : wave function symmetry ({0:}) different from requested symmetry ({1:})\n\n'.\
										format(symm.irrep_id2name(mol.groupname, wfnsym), symm.irrep_id2name(mol.groupname, calc.state['wfnsym'])))
				except Exception as err:
					sys.stderr.write(str(err))
					raise
		else:
			orbsym = np.zeros(hf.mo_occ.size, dtype=np.int)
		# debug print of orbital energies
		if mol.debug:
			if mol.symmetry:
				gpname = mol.symmetry
			else:
				gpname = 'C1'
			print('\n HF:  mo   symmetry    energy')
			for i in range(hf.mo_energy.size):
				print('     {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), hf.mo_energy[i]))
			print('\n')
		return hf, np.asscalar(e_hf), elec_dipole, occup, orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		if calc.model['type'] == 'occ':
			occ = np.where(hf.mo_occ == 2.)[0]
			virt = np.where(hf.mo_occ < 2.)[0]
		elif calc.model['type'] == 'virt':
			occ = np.where(hf.mo_occ > 0.)[0]
			virt = np.where(hf.mo_occ == 0.)[0]
		# nocc, nvirt, and norb
		nocc = len(occ)
		nvirt = len(virt)
		norb = nocc + nvirt
		return norb, nocc, nvirt


def active(mol, calc):
		""" set active space """
		# reference and expansion spaces
		if calc.model['type'] == 'occ':
			ref_space = np.array(range(mol.nocc, mol.norb))
			exp_space = np.array(range(mol.ncore, mol.nocc))
		elif calc.model['type'] == 'virt':
			ref_space = np.array(range(mol.ncore, mol.nocc))
			exp_space = np.array(range(mol.nocc, mol.norb))
		# hf reference model
		if calc.ref['method'] == 'hf':
			# no expansion space
			ne_act = (mol.nelec[0] - mol.ncore, mol.nelec[1] - mol.ncore)
			no_act = max(ne_act[0], ne_act[1])
			no_exp = 0
		# casci/casscf reference model
		elif calc.ref['method'] in ['casci','casscf']:
			if calc.ref['active'] == 'manual':
				# active electrons
				ne_act = calc.ref['nelec']
				# active orbs
				if isinstance(calc.ref['select'], dict):
					cas = mcscf.CASSCF(calc.hf, np.sum(list(calc.ref['select'].values())), ne_act)
					calc.ref['select'] = mcscf.caslst_by_irrep(cas, calc.mo, calc.ref['select'], base=0)
				calc.ref['select'] = np.asarray(calc.ref['select'])
				no_act = calc.ref['select'].size
				# expansion space orbs
				if calc.model['type'] == 'occ':
					no_exp = np.count_nonzero(calc.ref['select'] < mol.nocc)
				elif calc.model['type'] == 'virt':
					no_exp = np.count_nonzero(calc.ref['select'] >= mol.nocc)
				# sanity checks
				assert np.count_nonzero(calc.ref['select'] < mol.ncore) == 0
				assert float(ne_act[0] + ne_act[1]) <= np.sum(calc.hf.mo_occ[calc.ref['select']])
				# identical to hf ref?
				if no_exp == 0:
					try:
						raise RuntimeError('\nCAS Error: choice of CAS returns hf solution\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
			if mol.debug:
				print(' active: ne_act = {0:} , no_act = {1:} , no_exp = {2:}\n\n'.format(ne_act, no_act, no_exp))
		return ref_space, exp_space, no_exp, no_act, ne_act


def ref_mo(mol, calc, exp):
		""" determine reference mo coefficients """
		# sort mo coefficients
		if calc.no_exp == 0:
			mo = calc.mo
		elif calc.ref['method'] in ['casci','casscf']:
			if calc.ref['active'] == 'manual':
				# inactive region
				inact_elec = mol.nelectron - (calc.ne_act[0] + calc.ne_act[1])
				assert inact_elec % 2 == 0
				inact_orb = inact_elec // 2
				# divide into inactive-active-virtual
				idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['select']])
				mo = np.hstack((calc.mo[:, idx[:inact_orb]], calc.mo[:, calc.ref['select']], calc.mo[:, idx[inact_orb:]]))
				mo = np.asarray(mo, order='C')
				if mol.atom:
					calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo)
			# casscf mo
			if calc.ref['method'] == 'casscf':
				mo = _casscf(mol, calc, exp)
		return mo


def zero_energy(mol, calc, exp):
		""" calculate zeroth-order energy """
		# set core and cas spaces
		if calc.model['type'] == 'occ':
			exp.core_idx, exp.cas_idx = np.arange(mol.nocc), calc.ref_space
		elif calc.model['type'] == 'virt':
			exp.core_idx, exp.cas_idx = np.arange(mol.ncore), calc.ref_space
		# calculate energy
		zero = {}
		if mol.spin == 0:
			# no zeroth-order contributions
			if calc.target['energy']:
				zero['energy'] = 0.0
			if calc.target['excitation']:
				zero['excitation'] = 0.0
			if calc.target['dipole']:
				zero['dipole'] = np.zeros(3, dtype=np.float64)
			if calc.target['trans']:
				zero['trans'] = np.zeros(3, dtype=np.float64)
		else:
			# exp model
			res = main(mol, calc, exp, calc.model['method'])
			# e_ref
			if calc.target['energy']:
				zero['energy'] = res['energy']
				if calc.base['method'] is not None:
					res = main(mol, calc, exp, calc.base['method'])
					zero['energy'] -= res['energy']
			# excitation_ref
			if calc.target['excitation']:
				zero['excitation'] = res['excitation']
			# dipole_ref
			if calc.target['dipole']:
				zero['dipole'] = res['dipole']
			# trans_dipole_ref
			if calc.target['trans']:
				zero['trans'] = res['trans']
		return zero


def main(mol, calc, exp, method):
		""" main prop function """
		# fci calc
		if method == 'fci':
			res_tmp = _fci(mol, calc, exp)
		# cisd calc
		elif method == 'cisd':
			res_tmp = _ci(mol, calc, exp)
		# ccsd / ccsd(t) calc
		elif method in ['ccsd','ccsd(t)']:
			res_tmp = _cc(mol, calc, exp, method == 'ccsd(t)')
		# return correlation energy
		res = {}
		if calc.target['energy']:
			res['energy'] = res_tmp['energy']
		if calc.target['excitation']:
			res['excitation'] = res_tmp['excitation']
		# return first-order properties
		if calc.target['dipole']:
			if res_tmp['rdm1'] is None:
				res['dipole'] = None
			else:
				res['dipole'] = _dipole(mol, calc, exp, res_tmp['rdm1'])
		if calc.target['trans']:
			if res_tmp['t_rdm1'] is None:
				res['trans'] = None
			else:
				res['trans'] = _trans(mol, calc, exp, res_tmp['t_rdm1'], \
										res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])
		return res


def _dipole(mol, calc, exp, cas_rdm1, trans=False):
		""" calculate electronic (transition) dipole moment """
		# init (transition) rdm1
		if trans:
			rdm1 = np.zeros([mol.norb, mol.norb], dtype=np.float64)
		else:
			rdm1 = np.diag(calc.occup)
		# insert correlated subblock
		rdm1[exp.cas_idx[:, None], exp.cas_idx] = cas_rdm1
		# ao representation
		rdm1 = np.einsum('pi,ij,qj->pq', calc.mo, rdm1, calc.mo)
		# compute elec_dipole
		elec_dipole = np.einsum('xij,ji->x', mol.dipole, rdm1)
		# remove noise
		elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		# 'correlation' dipole
		if not trans:
			elec_dipole -= calc.prop['hf']['dipole']
		return elec_dipole


def _trans(mol, calc, exp, cas_t_rdm1, hf_weight_gs, hf_weight_ex):
		""" calculate electronic transition dipole moment """
		return _dipole(mol, calc, exp, cas_t_rdm1, True) * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol, calc, exp):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		exp.core_idx, exp.cas_idx = tools.core_cas(mol, exp, calc.exp_space)
		# init rdm1
		rdm1 = None
		# no base
		if calc.base['method'] is None:
			base = {'energy': 0.0}
		# cisd base
		elif calc.base['method'] == 'cisd':
			res = _ci(mol, calc, exp)
			base = {'energy': res['energy']}
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# ccsd / ccsd(t) base
		elif calc.base['method'] in ['ccsd','ccsd(t)']:
			res = _cc(mol, calc, exp, calc.base['method'] == 'ccsd(t)' and \
										(calc.orbs['occ'] == 'can' and calc.orbs['virt'] == 'can'))
			base = {'energy': res['energy']}
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# NOs
		if (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd') and rdm1 is None:
			res = _ci(mol, calc, exp)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		elif (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and rdm1 is None:
			res = _cc(mol, calc, exp, False)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		# occ-occ block (local or NOs)
		if calc.orbs['occ'] != 'can':
			if calc.orbs['occ'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
				calc.mo[:, mol.ncore:mol.nocc] = np.einsum('ip,pj->ij', calc.mo[:, mol.ncore:mol.nocc], no[:, ::-1])
			elif calc.orbs['occ'] == 'pm':
				calc.mo[:, mol.ncore:mol.nocc] = lo.PM(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] == 'fb':
				calc.mo[:, mol.ncore:mol.nocc] = lo.Boys(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] in ['ibo-1','ibo-2']:
				iao = lo.iao.iao(mol, calc.mo[:, mol.ncore:mol.nocc])
				if calc.orbs['occ'] == 'ibo-1':
					iao = lo.vec_lowdin(iao, calc.hf.get_ovlp())
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, calc.mo[:, mol.ncore:mol.nocc], iao)
				elif calc.orbs['occ'] == 'ibo-2':
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.PM(mol, calc.mo[:, mol.ncore:mol.nocc], iao).kernel()
		# virt-virt block (local or NOs)
		if calc.orbs['virt'] != 'can':
			if calc.orbs['virt'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
				calc.mo[:, mol.nocc:] = np.einsum('ip,pj->ij', calc.mo[:, mol.nocc:], no[:, ::-1])
			elif calc.orbs['virt'] == 'pm':
				calc.mo[:, mol.nocc:] = lo.PM(mol, calc.mo[:, mol.nocc:]).kernel()
			elif calc.orbs['virt'] == 'fb':
				calc.mo[:, mol.nocc:] = lo.Boys(mol, calc.mo[:, mol.nocc:]).kernel()
		# extra calculation for non-invariant ccsd(t)
		if calc.base['method'] == 'ccsd(t)' and (calc.orbs['occ'] != 'can' or calc.orbs['virt'] != 'can'):
			res = _cc(mol, calc, exp, True)
			base['energy'] = res['energy']
		return base


def _casscf(mol, calc, exp):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, calc.no_act, calc.ne_act)
		# fci solver
		cas.conv_tol = 1.0e-10
		cas.max_cycle_macro = 500
		# frozen (inactive)
		cas.frozen = mol.ncore
		# debug print
		if mol.debug: cas.verbose = 4
		# fcisolver
		if np.abs(calc.ne_act[0]-calc.ne_act[1]) == 0:
			if mol.symmetry:
				fcisolver = fci.direct_spin0_symm.FCI(mol)
			else:
				fcisolver = fci.direct_spin0.FCI(mol)
		else:
			if mol.symmetry:
				fcisolver = fci.direct_spin1_symm.FCI(mol)
			else:
				fcisolver = fci.direct_spin1.FCI(mol)
		# conv_tol
		fcisolver.conv_tol = max(calc.thres['init'], 1.0e-10)
		# orbital symmetry
		fcisolver.orbsym = calc.orbsym[mol.ncore:mol.ncore+calc.no_act]
		# wfnsym
		fcisolver.wfnsym = calc.ref['wfnsym'][0]
		# set solver
		cas.fcisolver = fcisolver
		# state-averaged calculation
		if len(calc.ref['wfnsym']) > 1:
			# weights
			weights = np.array((1 / len(calc.ref['wfnsym']),) * len(calc.ref['wfnsym']), dtype=np.float64)
			# are all states of same symmetry?
			if len(set(calc.ref['wfnsym'])) == 1:
				# state average
				cas.state_average_(weights)
			else:
				# nroots for first fcisolver
				fcisolver.nroots = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[0])
				# init list of fcisolvers
				fcisolvers = [fcisolver]
				# loop over symmetries
				for i in range(1, len(set(calc.ref['wfnsym']))):
					# copy fcisolver
					fcisolver_ = copy.copy(fcisolver)
					# wfnsym for fcisolver_
					fcisolver_.wfnsym = list(set(calc.ref['wfnsym']))[i]
					# nroots for fcisolver_
					fcisolver_.nroots = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[i])
					# append to fcisolvers
					fcisolvers.append(fcisolver_)
				# state average
				mcscf.state_average_mix_(cas, fcisolvers, weights)
		# hf starting guess
		if calc.ref['hf_guess']:
			na = fci.cistring.num_strings(calc.no_act, calc.ne_act[0])
			nb = fci.cistring.num_strings(calc.no_act, calc.ne_act[1])
			ci0 = np.zeros((na, nb))
			ci0[0, 0] = 1
		else:
			ci0 = None
		# run casscf calc
		cas.kernel(calc.mo, ci0=ci0)
		if len(calc.ref['wfnsym']) == 1:
			c = [cas.ci]
		else:
			c = cas.ci
		# multiplicity check
		for root in range(len(c)):
			s, mult = fcisolver.spin_square(c[root], calc.no_act, calc.ne_act)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(calc.ne_act[0]-calc.ne_act[1]) * 0.5
				cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.))
				# run casscf calc
				cas.kernel(calc.mo, ci0=ci0)
				if len(calc.ref['wfnsym']) == 1:
					c = [cas.ci]
				else:
					c = cas.ci
				# verify correct spin
				for root in range(len(c)):
					s, mult = fcisolver.spin_square(c[root], calc.no_act, calc.ne_act)
					assert np.abs((mol.spin + 1) - mult) < SPIN_TOL, ('\nCASSCF Error: spin contamination for root entry = {:}\n\n'
															'2*S + 1 = {:.6f}\n\n'). \
															format(root, mult)
		# convergence check
		if not cas.converged:
			try:
				raise RuntimeError('\nCASSCF Error: no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		return np.asarray(cas.mo_coeff, order='C')


def _fci(mol, calc, exp):
		""" fci calc """
		# electrons
		nelec = (mol.nelec[0] - len(exp.core_idx), mol.nelec[1] - len(exp.core_idx))
		# init fci solver
		if mol.spin == 0:
			solver = fci.direct_spin0_symm.FCI(mol)
		else:
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = max(calc.thres['init'], 1.0e-10)
		if calc.target['dipole'] or calc.target['trans']:
			solver.conv_tol *= 1.0e-04
			solver.lindep = solver.conv_tol * 1.0e-01
		solver.max_cycle = 5000
		solver.max_space = 25
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.state['wfnsym']
		# get integrals and core energy
		h1e, h2e = _prepare(mol, calc, exp)
		# orbital symmetry
		solver.orbsym = calc.orbsym[exp.cas_idx]
		# hf starting guess
		if calc.extra['hf_guess']:
			na = fci.cistring.num_strings(exp.cas_idx.size, nelec[0])
			nb = fci.cistring.num_strings(exp.cas_idx.size, nelec[1])
			ci0 = np.zeros((na, nb))
			ci0[0, 0] = 1
		else:
			ci0 = None
		# number of roots
		solver.nroots = calc.state['root'] + 1
		# interface
		def _fci_kernel():
				""" interface to solver.kernel """
				# perform calc
				e, c = solver.kernel(h1e, h2e, exp.cas_idx.size, nelec, ecore=mol.e_core, \
										orbsym=solver.orbsym, ci0=ci0)
				# collect results
				if solver.nroots == 1:
					return [e], [c]
				else:
					return [e[0], e[-1]], [c[0], c[-1]]
		# perform calc
		energy, civec = _fci_kernel()
		# multiplicity check
		for root in range(len(civec)):
			s, mult = solver.spin_square(civec[root], exp.cas_idx.size, nelec)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(nelec[0]-nelec[1]) * 0.5
				solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))
				# perform calc
				energy, civec = _fci_kernel()
				# verify correct spin
				for root in range(len(civec)):
					s, mult = solver.spin_square(civec[root], exp.cas_idx.size, nelec)
					assert np.abs((mol.spin + 1) - mult) < SPIN_TOL, ('\nFCI Error: spin contamination for root entry = {0:}\n\n'
															'2*S + 1 = {1:.6f}\n'
															'core_idx = {2:} , cas_idx = {3:}\n\n'). \
															format(root, mult, exp.core_idx, exp.cas_idx)
		# convergence check
		if solver.nroots == 1:
			assert solver.converged, ('FCI Error: state 0 not converged\n\n'
										'core_idx = {0:} , cas_idx = {1:}\n\n'). \
										format(exp.core_idx, exp.cas_idx)
		else:
			if calc.target['excitation']:
				for root in [0, solver.nroots-1]:
					assert solver.converged[root], ('FCI Error: state {0:} not converged\n\n'
											'core_idx = {1:} , cas_idx = {2:}\n\n'). \
											format(root, exp.core_idx, exp.cas_idx)
			else:
				assert solver.converged[solver.nroots-1], ('FCI Error: state {0:} not converged\n\n'
										'core_idx = {1:} , cas_idx = {2:}\n\n'). \
										format(solver.nroots-1, exp.core_idx, exp.cas_idx)
		res = {}
		# e_corr
		if calc.target['energy']:
			res['energy'] = energy[-1] - calc.prop['hf']['energy']
		if calc.target['excitation']:
			res['excitation'] = energy[-1] - energy[0]
		# fci rdm1 and t_rdm1
		if calc.target['dipole']:
			res['rdm1'] = solver.make_rdm1(civec[-1], exp.cas_idx.size, nelec)
		if calc.target['trans']:
			res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], exp.cas_idx.size, nelec)
			res['hf_weight'] = [civec[i][0, 0] for i in range(2)]
		return res


def _ci(mol, calc, exp):
		""" cisd calc """
		# get integrals
		h1e, h2e = _prepare(mol, calc, exp)
		mol_tmp = gto.M(verbose=1)
		mol_tmp.incore_anyway = mol.incore_anyway
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(len(exp.cas_idx)), mo_occ=calc.occup[exp.cas_idx])
		else:
			cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(len(exp.cas_idx)), np.eye(len(exp.cas_idx)))), \
									mo_occ=np.array((calc.occup[exp.cas_idx] > 0., calc.occup[exp.cas_idx] == 2.), dtype=np.double))
		# settings
		cisd.conv_tol = max(calc.thres['init'], 1.0e-10)
		cisd.max_cycle = 500
		cisd.max_space = 25
		eris = cisd.ao2mo()
		# calculate cisd energy
		cisd.kernel(eris=eris)
		# e_corr
		res = {'energy': cisd.e_corr}
		# rdm1
		if exp.order == 0 and (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd'):
			res['rdm1'] = cisd.make_rdm1()
		return res


def _cc(mol, calc, exp, pt=False):
		""" ccsd / ccsd(t) calc """
		# get integrals
		h1e, h2e = _prepare(mol, calc, exp)
		mol_tmp = gto.M(verbose=1)
		mol_tmp.incore_anyway = mol.incore_anyway
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(len(exp.cas_idx)), mo_occ=calc.occup[exp.cas_idx])
		else:
			ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(len(exp.cas_idx)), np.eye(len(exp.cas_idx)))), \
									mo_occ=np.array((calc.occup[exp.cas_idx] > 0., calc.occup[exp.cas_idx] == 2.), dtype=np.double))
		# settings
		ccsd.conv_tol = max(calc.thres['init'], 1.0e-10)
		if exp.order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and not pt:
			ccsd.conv_tol_normt = ccsd.conv_tol
		ccsd.max_cycle = 500
		# avoid async function execution if requested
		ccsd.async_io = calc.misc['async']
		# avoid I/O if not async
		if not calc.misc['async']: ccsd.incore_complete = True
		eris = ccsd.ao2mo()
		# calculate ccsd energy
		for i in list(range(0, 12, 2)):
			ccsd.diis_start_cycle = i
			try:
				ccsd.kernel(eris=eris)
			except sp.linalg.LinAlgError:
				pass
			if ccsd.converged: break
		# convergence check
		if not ccsd.converged:
			try:
				raise RuntimeError('\nCCSD Error: no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		res = {'energy': ccsd.e_corr}
		# rdm1
		if exp.order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and not pt:
			ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
			res['rdm1'] = ccsd.make_rdm1()
		# calculate (t) correction
		if pt:
			if np.amin(calc.occup[exp.cas_idx]) == 1.0:
				if len(np.where(calc.occup[exp.cas_idx] == 1.)[0]) >= 3:
					res['energy'] += ccsd.ccsd_t(eris=eris)
			else:
				res['energy'] += ccsd.ccsd_t(eris=eris)
		return res


def _prepare(mol, calc, exp):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if mol.e_core is None or exp.model['type'] == 'occ':
			if exp.core_idx.size > 0:
				core_dm = np.einsum('ip,jp->ij', calc.mo[:, exp.core_idx], calc.mo[:, exp.core_idx]) * 2
				vj, vk = scf.hf.dot_eri_dm(mol.eri, core_dm)
				mol.core_vhf = vj - vk * .5
				mol.e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
				mol.e_core += np.einsum('ij,ji', core_dm, mol.core_vhf) * .5
			else:
				mol.e_core = mol.energy_nuc()
				mol.core_vhf = 0
		h1e_cas = np.einsum('pi,pq,qj->ij', calc.mo[:, exp.cas_idx], mol.hcore + mol.core_vhf, calc.mo[:, exp.cas_idx])
		h2e_cas = ao2mo.incore.full(mol.eri, calc.mo[:, exp.cas_idx])
		return h1e_cas, h2e_cas


