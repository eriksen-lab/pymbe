#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
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
from pyscf import tools as pyscf_tools
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

import parallel
import tools


SPIN_TOL = 1.0e-05
DIPOLE_TOL = 1.0e-14


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
		if calc.target in ['dipole', 'trans']:
			with mol.with_common_origin([0.0, 0.0, 0.0]):
				dipole = mol.intor_symmetric('int1e_r', comp=3)
		else:
			dipole = None
		return hcore, eri, dipole


def _hubbard_h1(mol):
		""" set hubbard hopping hamiltonian """
		# dimension
		if 1 in mol.matrix:
			dim = 1
		else:
			dim = 2
		# init h1
		h1 = np.zeros([mol.nsites] * 2, dtype=np.float64)
		# 1d
		if dim == 1:
			# adjacent neighbours
			for i in range(mol.nsites-1):
				h1[i, i+1] = h1[i+1, i] = -1.0
			# pbc
			if mol.pbc:
				h1[-1, 0] = h1[0, -1] = -1.0
		# 2d
		elif dim == 2:
			nx, ny = mol.matrix[0], mol.matrix[1]
			# init
			for site_1 in range(mol.nsites):
				site_1_xy = tools.mat_indx(site_1, nx, ny)
				nbrs = tools.near_nbrs(site_1_xy, nx, ny)
				for site_2 in range(site_1):
					site_2_xy = tools.mat_indx(site_2, nx, ny)
					if site_2_xy in nbrs:
						h1[site_1, site_2] = h1[site_2, site_1] = -1.0
			# pbc
			if mol.pbc:
				# sideways
				for i in range(ny):
					h1[i, ny * (nx - 1) + i] = h1[ny * (nx - 1) + i, i] = -1.0
				# up-down
				for i in range(nx):
					h1[i * ny, i * ny + (ny - 1)] = h1[i * ny + (ny - 1), i * ny] = -1.0
		return h1


def _hubbard_eri(mol):
		""" set hubbard two-electron hamiltonian """
		# init eri
		eri = np.zeros([mol.nsites] * 4, dtype=np.float64)
		for i in range(mol.nsites):
			eri[i,i,i,i] = mol.u
		return ao2mo.restore(8, eri, mol.nsites)


def hf(mol, calc):
		""" hartree-fock calculation """
		# perform restricted hf calc
		mol_hf = mol.copy()
		mol_hf.build(0, 0, symmetry = mol.hf_sym)
		hf = scf.RHF(mol_hf)
		# debug print
		if mol.debug >= 1:
			hf.verbose = 4
		hf.init_guess = mol.hf_init_guess
		hf.conv_tol = 1.0e-09
		hf.max_cycle = 1000
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
			except sp.linalg.LinAlgError:
				pass
			if hf.converged:
				break
		# convergence check
		tools.assertion(hf.converged, 'HF error: no convergence')
		# dipole moment
		if calc.target == 'dipole':
			dm = hf.make_rdm1()
			elec_dipole = np.einsum('xij,ji->x', mol.dipole, dm)
			elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0.0 for i in range(elec_dipole.size)])
		else:
			elec_dipole = None
		# determine dimensions
		norb, nocc, nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		if mol.atom:
			orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
		else:
			orbsym = np.zeros(hf.mo_energy.size, dtype=np.int)
		# debug print of orbital energies
		if mol.debug >= 1:
			if mol.symmetry:
				gpname = mol.symmetry
			else:
				gpname = 'C1'
			print('\n HF:  mo   symmetry    energy')
			for i in range(hf.mo_energy.size):
				print('     {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), hf.mo_energy[i]))
			print('\n')
		return nocc, nvirt, norb, hf, np.asscalar(e_hf), elec_dipole, occup, orbsym, hf.mo_energy, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		occ = np.where(hf.mo_occ > 0.)[0]
		virt = np.where(hf.mo_occ == 0.)[0]
		# nocc, nvirt, and norb
		nocc = occ.size
		nvirt = virt.size
		norb = nocc + nvirt
		return norb, nocc, nvirt


def ref_mo(mol, calc):
		""" determine reference mo coefficients """
		# transform mo coefficients 
		if calc.orbs['type'] in ['ccsd', 'ccsd(t)', 'local']:
			# set core and cas spaces
			core_idx, cas_idx = tools.core_cas(mol, np.arange(mol.ncore, mol.nocc), np.arange(mol.nocc, mol.norb))
			# NOs
			if calc.orbs['type'] in ['ccsd', 'ccsd(t)']:
				# get rdm1
				rdm1 = _cc(mol, calc, calc.mo_coeff, core_idx, cas_idx, calc.orbs['type'], True)
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
				mo_coeff = _nat_orbs(mol, rdm1, calc.mo_coeff, calc.orbsym, cas_idx)
			# pipek-mezey localized orbitals
			elif calc.orbs['type'] == 'local':
				mo_coeff = _loc_orbs(mol, calc.mo_coeff, cas_idx)
			mo_energy = calc.mo_energy
		else:
			mo_coeff = calc.mo_coeff
			mo_energy = calc.mo_energy
		# active orbs
		calc.ref['select'] = np.asarray(calc.ref['select'], dtype=np.int32)
		# electrons
		nelec = (np.count_nonzero(calc.occup[calc.ref['select']] > 0.), \
					np.count_nonzero(calc.occup[calc.ref['select']] > 1.))
		# inactive orbitals
		inact_elec = mol.nelectron - (nelec[0] + nelec[1])
		tools.assertion(inact_elec % 2 == 0, 'odd number of inactive electrons')
		inact_orbs = inact_elec // 2
		# active orbitals
		act_orbs = calc.ref['select'].size
		# virtual orbitals
		virt_orbs = mol.norb - inact_orbs - act_orbs
		# divide into inactive-active-virtual
		idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['select']])
		if act_orbs > 0:
			mo_coeff = np.concatenate((mo_coeff[:, idx[:inact_orbs]], \
										mo_coeff[:, calc.ref['select']], \
										mo_coeff[:, idx[inact_orbs:]]), axis=1)
			if mol.atom:
				calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
		# reference and expansion spaces
		ref_space = np.arange(inact_orbs, inact_orbs+act_orbs)
		exp_space = np.append(np.arange(mol.ncore, inact_orbs), np.arange(inact_orbs+act_orbs, mol.norb))
		# casci or casscf
		if calc.ref['method'] == 'casci':
			if act_orbs > 0:
				mo_energy = np.concatenate((mo_energy[idx[:inact_orbs]], \
											mo_energy[calc.ref['select']], \
											mo_energy[idx[inact_orbs:]]))
		elif calc.ref['method'] == 'casscf':
			tools.assertion(np.count_nonzero(calc.occup[calc.ref['select']] == 2.) != 0, \
							'no occupied orbitals in CASSCF calculation')
			tools.assertion(np.count_nonzero(calc.occup[calc.ref['select']] == 0.) != 0, \
							'no virtual orbitals in CASSCF calculation')
			# casscf quantities
			mo_energy, mo_coeff = _casscf(mol, calc, mo_coeff, ref_space, nelec)
		if mol.debug >= 1:
			print('\n reference nelec  = {:}'.format(nelec))
			print(' reference space  = {:}'.format(ref_space))
			print(' expansion space  = {:}\n'.format(exp_space))
		return mo_energy, np.asarray(mo_coeff, order='C'), nelec, ref_space, exp_space


def dyn_orbs(mpi, mol, calc, exp):
		""" update dynamic orbitals """
		if mpi.master:
			calc.mo_coeff = _nat_orbs(mol, exp.rdm1['tot'], calc.mo_coeff, calc.orbsym)
		parallel.mo(mpi, calc.mo_coeff)
		if mol.atom:
			calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)


def _nat_orbs(mol, rdm1, mo_coeff_in, orbsym, space):
		""" diagonalize rdm1 to compute natural orbitals """
		# copy MOs
		mo_coeff_out = np.copy(mo_coeff_in)
		# occ-occ part
		occ_idx = space[space < mol.nocc]
		occup, no = symm.eigh(rdm1[:occ_idx.size, :occ_idx.size], orbsym[occ_idx])
		mo_coeff_out[:, occ_idx] = np.einsum('ip,pj->ij', mo_coeff_in[:, occ_idx], no[:, ::-1])
		# virt-virt part
		virt_idx = space[space >= mol.nocc]
		occup, no = symm.eigh(rdm1[-virt_idx.size:, -virt_idx.size:], orbsym[virt_idx])
		mo_coeff_out[:, virt_idx] = np.einsum('ip,pj->ij', mo_coeff_in[:, virt_idx], no[:, ::-1])
		return mo_coeff_out


def _loc_orbs(mol, mo_coeff_in, space):
		""" compute pipek-mezey localized orbitals """
		# copy MOs
		mo_coeff_out = np.copy(mo_coeff_in)
		# occ-occ part
		occ_idx = space[space < mol.nocc]
		if mol.atom:
			mo_coeff_out[:, occ_idx] = lo.PM(mol, mo_coeff_in[:, occ_idx]).kernel()
		else:
			mo_coeff_out[:, occ_idx] = tools.hubbard_PM(mol, mo_coeff_in[:, occ_idx]).kernel()
		# virt-virt part
		virt_idx = space[space >= mol.nocc]
		if mol.atom:
			mo_coeff_out[:, virt_idx] = lo.PM(mol, mo_coeff_in[:, virt_idx]).kernel()
		else:
			mo_coeff_out[:, virt_idx] = tools.hubbard_PM(mol, mo_coeff_in[:, virt_idx]).kernel()
		return mo_coeff_out


def ref_prop(mol, calc, exp):
		""" calculate reference space properties """
		# generate input
		exp.core_idx, exp.cas_idx = tools.core_cas(mol, calc.ref_space, np.array([], dtype=np.int32))
		# nelec
		nelec = np.asarray((np.count_nonzero(calc.occup[exp.cas_idx] > 0.), \
							np.count_nonzero(calc.occup[exp.cas_idx] > 1.)), dtype=np.int32)
		# reference space prop
		if np.any(calc.occup[calc.ref_space] == 2.) and np.any(calc.occup[calc.ref_space] < 2.):
			# exp model
			ref = main(mol, calc, exp, calc.model['method'], nelec)
			if calc.base['method'] is not None:
				# base model
				ref -= main(mol, calc, exp, calc.base['method'], nelec)
		else:
			# no correlation in expansion reference space
			if calc.target in ['energy', 'excitation']:
				ref = 0.0
			else:
				ref = np.zeros(3, dtype=np.float64)
		return ref


def main(mol, calc, exp, method, nelec):
		""" main prop function """
		# no occupied or no virtuals
		if np.all(calc.occup[exp.cas_idx] == 2.) or np.all(calc.occup[exp.cas_idx] == 0.):
			if calc.target in ['energy', 'excitation']:
				return 0.0
			else:
				return np.zeros(3, dtype=np.float64)
		if method in ['ccsd','ccsd(t)']:
			# ccsd / ccsd(t) calc
			res = _cc(mol, calc, calc.mo_coeff, exp.core_idx, exp.cas_idx, method)
			rdm1 = None
		elif method == 'fci':
			# fci calc
			res_tmp = _fci(mol, calc, calc.mo_coeff, exp.core_idx, exp.cas_idx, nelec)
			if calc.target in ['energy', 'excitation']:
				res = res_tmp[calc.target]
			elif calc.target == 'dipole':
				res = _dipole(mol, calc, exp, res_tmp['rdm1'])
			elif calc.target == 'trans':
				res = _trans(mol, calc, exp, res_tmp['t_rdm1'], \
								res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])
			if calc.orbs['type'] == 'dynamic':
				rdm1 = res_tmp['rdm1']
			else:
				rdm1 = None
		return res, rdm1


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
		rdm1 = np.einsum('pi,ij,qj->pq', calc.mo_coeff, rdm1, calc.mo_coeff)
		# compute elec_dipole
		elec_dipole = np.einsum('xij,ji->x', mol.dipole, rdm1)
		# remove noise
		elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0.0 for i in range(elec_dipole.size)])
		# 'correlation' dipole
		if not trans:
			elec_dipole -= calc.prop['hf']['dipole']
		return elec_dipole


def _trans(mol, calc, exp, cas_t_rdm1, hf_weight_gs, hf_weight_ex):
		""" calculate electronic transition dipole moment """
		return _dipole(mol, calc, exp, cas_t_rdm1, True) * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol, calc):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		core_idx, cas_idx = tools.core_cas(mol, np.arange(mol.ncore, mol.nocc), np.arange(mol.nocc, mol.norb))
		# no base
		if calc.base['method'] is None:
			e_base = 0.0
		# ccsd / ccsd(t) base
		elif calc.base['method'] in ['ccsd','ccsd(t)']:
			e_base = _cc(mol, calc, calc.mo_coeff, core_idx, cas_idx, calc.base['method'])
		return e_base


def _casscf(mol, calc, mo_coeff, ref_space, nelec):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, ref_space.size, nelec)
		# fci solver
		cas.conv_tol = 1.0e-10
		cas.max_cycle_macro = 500
		# frozen (inactive)
		cas.frozen = mol.ncore
		# debug print
		if mol.debug >= 1:
			cas.verbose = 4
		# fcisolver
		if calc.model['solver'] == 'pyscf_spin0':
			fcisolver = fci.direct_spin0_symm.FCI(mol)
		elif calc.model['solver'] == 'pyscf_spin1':
			fcisolver = fci.direct_spin1_symm.FCI(mol)
		# conv_tol
		fcisolver.conv_tol = max(calc.thres['init'], 1.0e-10)
		# orbital symmetry
		fcisolver.orbsym = calc.orbsym[ref_space]
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
			na = fci.cistring.num_strings(ref_space.size, nelec[0])
			nb = fci.cistring.num_strings(ref_space.size, nelec[1])
			ci0 = np.zeros((na, nb))
			ci0[0, 0] = 1
		else:
			ci0 = None
		# run casscf calc
		cas.kernel(mo_coeff, ci0=ci0)
		if len(calc.ref['wfnsym']) == 1:
			c = [cas.ci]
		else:
			c = cas.ci
		# multiplicity check
		for root in range(len(c)):
			s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(nelec[0]-nelec[1]) * 0.5
				cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.))
				# run casscf calc
				cas.kernel(mo_coeff, ci0=ci0)
				if len(calc.ref['wfnsym']) == 1:
					c = [cas.ci]
				else:
					c = cas.ci
				# verify correct spin
				for root in range(len(c)):
					s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
					tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
									'spin contamination for root entry = {:} , 2*S + 1 = {:.6f}'. \
										format(root, mult))
		# convergence check
		tools.assertion(cas.converged, 'CASSCF error: no convergence')
		# debug print of orbital energies
		if mol.atom:
			orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, cas.mo_coeff)
		else:
			orbsym = np.zeros(cas.mo_energy.size, dtype=np.int)
		if mol.debug >= 1:
			if mol.symmetry:
				gpname = mol.symmetry
			else:
				gpname = 'C1'
			print('\n CASSCF:  mo   symmetry    energy')
			for i in range(cas.mo_energy.size):
				print('         {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), cas.mo_energy[i]))
			print('\n')
		return cas.mo_energy, np.asarray(cas.mo_coeff, order='C')


def _fci(mol, calc, mo_coeff, core_idx, cas_idx, nelec):
		""" fci calc """
		# init fci solver
		if calc.model['solver'] == 'pyscf_spin0':
			solver = fci.direct_spin0_symm.FCI(mol)
		elif calc.model['solver'] == 'pyscf_spin1':
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = max(calc.thres['init'], 1.0e-10)
		if calc.target in ['dipole', 'trans']:
			solver.conv_tol *= 1.0e-04
			solver.lindep = solver.conv_tol * 1.0e-01
		solver.max_cycle = 5000
		solver.max_space = 25
		solver.davidson_only = True
		solver.pspace_size = 0
		# debug print
		if mol.debug >= 3:
			solver.verbose = 10
		# wfnsym
		solver.wfnsym = calc.state['wfnsym']
		# get integrals and core energy
		h1e, h2e, e_core = _prepare(mol, mo_coeff, core_idx, cas_idx)
		# orbital symmetry
		solver.orbsym = calc.orbsym[cas_idx]
		# hf starting guess
		if calc.extra['hf_guess']:
			na = fci.cistring.num_strings(cas_idx.size, nelec[0])
			nb = fci.cistring.num_strings(cas_idx.size, nelec[1])
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
				e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, \
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
			s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(nelec[0]-nelec[1]) * 0.5
				solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))
				# perform calc
				energy, civec = _fci_kernel()
				# verify correct spin
				for root in range(len(civec)):
					s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
					tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
									'spin contamination for root entry = {:} , 2*S + 1 = {:.6f} , '
									'core_idx = {:} , cas_idx = {:}'. \
										format(root, mult, core_idx, cas_idx))
		# convergence check
		if solver.nroots == 1:
			tools.assertion(solver.converged, 'state 0 not converged , core_idx = {:} , cas_idx = {:}'. \
								format(core_idx, cas_idx))
		else:
			if calc.target == 'excitation':
				for root in [0, solver.nroots-1]:
					tools.assertion(solver.converged[root], 'state {:} not converged , core_idx = {:} , cas_idx = {:}'. \
										format(root, core_idx, cas_idx))
			else:
				tools.assertion(solver.converged[solver.nroots-1], 'state {:} not converged , core_idx = {:} , cas_idx = {:}'. \
										format(solver.nroots-1, core_idx, cas_idx))
		res = {}
		# e_corr
		if calc.target == 'energy':
			res['energy'] = energy[-1] - calc.prop['hf']['energy']
		if calc.target == 'excitation':
			res['excitation'] = energy[-1] - energy[0]
		# fci rdm1 and t_rdm1
		if calc.target == 'dipole' or calc.orbs['type'] == 'dynamic':
			res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, nelec)
		if calc.target == 'trans':
			res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, nelec)
			res['hf_weight'] = [civec[i][0, 0] for i in range(2)]
		return res


def _cc(mol, calc, mo_coeff, core_idx, cas_idx, method, rdm=False):
		""" ccsd / ccsd(t) calc """
		# get integrals
		h1e, h2e, e_core = _prepare(mol, mo_coeff, core_idx, cas_idx)
		mol_tmp = gto.M(verbose=0)
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
			ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(cas_idx.size), mo_occ=calc.occup[cas_idx])
		else:
			ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))), \
									mo_occ=np.array((calc.occup[cas_idx] > 0., calc.occup[cas_idx] == 2.), dtype=np.double))
		# settings
		ccsd.conv_tol = max(calc.thres['init'], 1.0e-10)
		if rdm:
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
			if ccsd.converged:
				break
		# convergence check
		tools.assertion(ccsd.converged, 'CCSD error: no convergence')
		# e_corr
		e_cc = ccsd.e_corr
		# calculate (t) correction
		if method == 'ccsd(t)':
			if np.amin(calc.occup[cas_idx]) == 1.0:
				if np.where(calc.occup[cas_idx] == 1.)[0].size >= 3:
					e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)
			else:
				e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)
		# rdm1
		if not rdm:
			return e_cc
		else:
			if method == 'ccsd':
				ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
				rdm1 = ccsd.make_rdm1()
			elif method == 'ccsd(t)':
				l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
				rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
			return rdm1


def _prepare(mol, mo_coeff, core_idx, cas_idx):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if core_idx.size > 0:
			core_dm = np.einsum('ip,jp->ij', mo_coeff[:, core_idx], mo_coeff[:, core_idx]) * 2
			vj, vk = scf.hf.dot_eri_dm(mol.eri, core_dm)
			core_vhf = vj - vk * .5
			e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
			e_core += np.einsum('ij,ji', core_dm, core_vhf) * .5
		else:
			core_vhf = 0
			e_core = mol.energy_nuc()
		h1e_cas = np.einsum('pi,pq,qj->ij', mo_coeff[:, cas_idx], mol.hcore + core_vhf, mo_coeff[:, cas_idx])
		h2e_cas = ao2mo.incore.full(mol.eri, mo_coeff[:, cas_idx])
		return h1e_cas, h2e_cas, e_core


