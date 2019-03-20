#!/usr/bin/env python
# -*- coding: utf-8 -*

""" parallel.py: mpi class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from pyscf import symm

import restart


class MPICls(object):
		""" mpi parameters """
		def __init__(self):
				""" init parameters """
				self.comm = MPI.COMM_WORLD
				self.size = self.comm.Get_size()
				self.rank = self.comm.Get_rank()
				self.master = (self.rank == 0)
				self.slave = not self.master
				self.host = MPI.Get_processor_name()
				self.stat = MPI.Status()


def mol(mpi, mol):
		""" bcast mol info """
		if mpi.master:
			info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, 'e_core': mol.e_core, \
					'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
					'cart': mol.cart, 'unit': mol.unit, 'frozen': mol.frozen, 'debug': mol.debug}
			if not mol.atom:
				info['u'] = mol.u
				info['n'] = mol.n
				info['matrix'] = mol.matrix
				info['nsites'] = mol.nsites
				info['pbc'] = mol.pbc
				info['nelec'] = mol.nelectron
			mpi.comm.bcast(info, root=0)
		else:
			info = mpi.comm.bcast(None, root=0)
			mol.atom = info['atom']; mol.charge = info['charge']
			mol.spin = info['spin']; mol.e_core = info['e_core']
			mol.symmetry = info['symmetry']; mol.irrep_nelec = info['irrep_nelec']
			mol.basis = info['basis']; mol.cart = info['cart']
			mol.unit = info['unit']; mol.frozen = info['frozen']
			mol.debug = info['debug']
			if not mol.atom:
				mol.u = info['u']; mol.n = info['n']
				mol.matrix = info['matrix']; mol.nsites = info['nsites'] 
				mol.pbc = info['pbc']; mol.nelectron = info['nelec']


def calc(mpi, calc):
		""" bcast calc info """
		if mpi.master:
			info = {'model': calc.model, 'target': calc.target, \
					'base': calc.base, \
					'thres': calc.thres, 'prot': calc.prot, \
					'state': calc.state, 'extra': calc.extra, \
					'misc': calc.misc, 'mpi': calc.mpi, \
					'orbs': calc.orbs, 'restart': calc.restart}
			mpi.comm.bcast(info, root=0)
		else:
			info = mpi.comm.bcast(None, root=0)
			calc.model = info['model']; calc.target = info['target']
			calc.base = info['base']
			calc.thres = info['thres']; calc.prot = info['prot']
			calc.state = info['state']; calc.extra = info['extra']
			calc.misc = info['misc']; calc.mpi = info['mpi']
			calc.orbs = info['orbs']; calc.restart = info['restart']


def fund(mpi, mol, calc):
		""" bcast fundamental info """
		if mpi.master:
			info = {'prop': calc.prop, \
						'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, \
						'ref_space': calc.ref_space, 'exp_space': calc.exp_space, \
						'occup': calc.occup, 'mo_energy': calc.mo_energy}
			mpi.comm.bcast(info, root=0)
			# bcast mo coefficients
			mo_coeff(mpi, calc.mo_coeff)
		else:
			info = mpi.comm.bcast(None, root=0)
			calc.prop = info['prop']
			mol.norb = info['norb']; mol.nocc = info['nocc']; mol.nvirt = info['nvirt']
			calc.ref_space = info['ref_space']; calc.exp_space = info['exp_space']
			calc.occup = info['occup']; calc.mo_energy = info['mo_energy']
			# receive mo coefficients
			calc.mo_coeff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
			mo_coeff(mpi, calc.mo_coeff)
		if mol.atom:
			calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
		else:
			calc.orbsym = np.zeros(mol.norb, dtype=np.int)


def exp(mpi, calc, exp):
		""" bcast exp info """
		if mpi.master:
			# collect info
			info = {'len_tup': [exp.tuples[i].shape[0] for i in range(len(exp.tuples))]}
			info['len_prop'] = [exp.prop[calc.target]['inc'][i].shape[0] for i in range(len(exp.prop[calc.target]['inc']))]
			# bcast info
			mpi.comm.bcast(info, root=0)
			# bcast tuples and hashes
			for i in range(1, len(exp.tuples)):
				mpi.comm.Bcast([exp.tuples[i], MPI.INT], root=0)
			for i in range(1, len(exp.hashes)):
				mpi.comm.Bcast([exp.hashes[i], MPI.INT], root=0)
			# bcast increments
			for i in range(len(exp.prop[calc.target]['inc'])):
				mpi.comm.Bcast([exp.prop[calc.target]['inc'][i], MPI.DOUBLE], root=0)
		else:
			# receive info
			info = mpi.comm.bcast(None, root=0)
			# receive tuples and hashes
			for i in range(1, len(info['len_tup'])):
				buff = np.empty([info['len_tup'][i], i+1], dtype=np.int32)
				mpi.comm.Bcast([buff, MPI.INT], root=0)
				exp.tuples.append(buff)
			for i in range(1, len(info['len_tup'])):
				buff = np.empty(info['len_tup'][i], dtype=np.int64)
				mpi.comm.Bcast([buff, MPI.INT], root=0)
				exp.hashes.append(buff)
			# receive increments
			if calc.target in ['energy', 'excitation']:
				for i in range(len(info['len_prop'])):
					buff = np.zeros(info['len_prop'][i], dtype=np.float64)
					mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
					exp.prop[calc.target]['inc'].append(buff)
			else:
				for i in range(len(info['len_prop'])):
					buff = np.zeros([info['len_prop'][i], 3], dtype=np.float64)
					mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
					exp.prop[calc.target]['inc'].append(buff)


def mo_coeff(mpi, mo_coeff):
		""" Bcast mo coefficients """
		mpi.comm.Bcast([mo_coeff, MPI.DOUBLE], root=0)


def rdm1(mpi, rdm1):
		""" ALlreduce rdm1s """
		mpi.comm.Allreduce(MPI.IN_PLACE, rdm1, op=MPI.SUM)
#		if mpi.master:
#			mpi.comm.Reduce([np.empty_like(rdm1), MPI.DOUBLE], [rdm1, MPI.DOUBLE], op=MPI.SUM, root=0)
#		else:
#			mpi.comm.Reduce([rdm1, MPI.DOUBLE], [np.empty_like(rdm1), MPI.DOUBLE], op=MPI.SUM, root=0)


def mbe(mpi, prop, ndets):
		""" Allreduce properties and number of determinants """
		mpi.comm.Allreduce(MPI.IN_PLACE, prop, op=MPI.SUM)
		mpi.comm.Allreduce(MPI.IN_PLACE, ndets, op=MPI.SUM)


def screen(mpi, child_tup, child_hash, order):
		""" Allgatherv tuples / hashes """
		# allgatherv tuples
		child_tup = np.asarray(child_tup, dtype=np.int32)
		recv_counts = np.array(mpi.comm.allgather(child_tup.size), dtype=np.int32)
		tuples = np.empty(np.sum(recv_counts, dtype=np.int64), dtype=np.int32)
		mpi.comm.Allgatherv(child_tup, [tuples, recv_counts])
		tuples = tuples.reshape(-1, order+1)
		if tuples.shape[0] > 0:
			# allgatherv hashes
			child_hash = np.asarray(child_hash, dtype=np.int64)
			recv_counts = np.array(mpi.comm.allgather(child_hash.size), dtype=np.int64)
			hashes = np.empty(np.sum(recv_counts, dtype=np.int64), dtype=np.int64)
			mpi.comm.Allgatherv(child_hash, [hashes, recv_counts])
			# sort wrt hashes
			tuples = tuples[hashes.argsort()]
			# sort hashes
			hashes.sort()
		else:
			hashes = np.array([], dtype=np.int64)
		return tuples, hashes


def abort():
		""" abort calculation """
		MPI.COMM_WORLD.Abort()


def final(mpi):
		""" terminate calculation """
		if mpi.master:
			restart.rm()
			mpi.comm.bcast({'task': 'exit'}, root=0)
		mpi.comm.Barrier()
		MPI.Finalize()


