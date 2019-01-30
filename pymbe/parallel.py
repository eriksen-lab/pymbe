#!/usr/bin/env python
# -*- coding: utf-8 -*

""" parallel.py: mpi class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import sys
import traceback
from pyscf import symm

import tools
import restart


class MPICls():
		""" mpi parameters """
		def __init__(self):
				""" init parameters """
				self.global_comm = MPI.COMM_WORLD
				self.global_group = self.global_comm.Get_group()
				self.parallel = (self.global_comm.Get_size() > 1)
				self.global_size = self.global_comm.Get_size()
				self.global_rank = self.global_comm.Get_rank()
				self.global_master = (self.global_rank == 0)
				self.host = MPI.Get_processor_name()
				self.stat = MPI.Status()
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(variant, value, trace):
					""" custom mpi exception hook """
					if not issubclass(variant, OSError):
						print('\n-- Error information --')
						print('\ntype:\n\n  {0:}'.format(variant))
						print('\nvalue:\n\n  {0:}'.format(value))
						print('\ntraceback:\n\n{0:}'.format(''.join(traceback.format_tb(trace))))
					sys_excepthook(variant, value, trace)
					self.global_comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook


def set_mpi(mpi, calc):
		""" set mpi info """
		# now set info
		if not mpi.parallel:
			mpi.prim_master = True
			mpi.global_rank = mpi.local_rank = 0
		else:
			if calc.mpi['masters'] == 1:
				mpi.master_comm = mpi.local_comm = mpi.global_comm
				mpi.local_size = mpi.global_size
				mpi.local_rank = mpi.global_rank
				mpi.local_master = False
				mpi.prim_master = mpi.global_master
			else:
				# array of ranks (global master excluded)
				ranks = np.arange(1, mpi.global_size)
				# split into local groups and add global master
				groups = np.array_split(ranks, calc.mpi['masters']-1)
				groups = [np.array([0])] + groups
				# extract local master indices and append to global master index
				masters = [groups[i][0] for i in range(len(groups))]
				# define local masters (global master excluded)
				mpi.local_master = (mpi.global_rank in masters[1:])
				# define primary local master
				mpi.prim_master = (mpi.global_rank == 1)
				# set master group and intracomm
				mpi.master_group = mpi.global_group.Incl(masters)
				mpi.master_comm = mpi.global_comm.Create(mpi.master_group)
				# set local master group and intracomm
				mpi.local_master_group = mpi.master_group.Excl([0])
				mpi.local_master_comm = mpi.global_comm.Create(mpi.local_master_group)
				# set local intracomm based on color
				for i in range(len(groups)):
					if mpi.global_rank in groups[i]: color = i
				mpi.local_comm = mpi.global_comm.Split(color)
				# determine size of local group and recover rank
				mpi.local_size = mpi.local_comm.Get_size()
				mpi.local_rank = mpi.local_comm.Get_rank()
			# define slave
			if not mpi.global_master and not mpi.local_master:
				mpi.slave = True
			else:
				mpi.slave = False


def mol(mpi, mol):
		""" bcast mol info """
		if mpi.parallel:
			if mpi.global_master:
				info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, 'e_core': mol.e_core, \
						'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
						'cart': mol.cart, 'unit': mol.unit, 'frozen': mol.frozen, 'debug': mol.debug}
				if not mol.atom:
					info['t'] = mol.t
					info['u'] = mol.u
					info['dim'] = mol.dim
					info['nsites'] = mol.nsites
					info['pbc'] = mol.pbc
					info['nelec'] = mol.nelectron
				mpi.global_comm.bcast(info, root=0)
			else:
				info = mpi.global_comm.bcast(None, root=0)
				mol.atom = info['atom']; mol.charge = info['charge']
				mol.spin = info['spin']; mol.e_core = info['e_core']
				mol.symmetry = info['symmetry']; mol.irrep_nelec = info['irrep_nelec']
				mol.basis = info['basis']; mol.cart = info['cart']
				mol.unit = info['unit']; mol.frozen = info['frozen']
				mol.debug = info['debug']
				if not mol.atom:
					mol.t = info['t']; mol.u = info['u']; mol.dim = info['dim']
					mol.nsites = info['nsites']; mol.pbc = info['pbc']; mol.nelectron = info['nelec']


def calc(mpi, calc):
		""" bcast calc info """
		if mpi.parallel:
			if mpi.global_master:
				info = {'model': calc.model, 'target': calc.target, \
						'base': calc.base, \
						'thres': calc.thres, 'prot': calc.prot, \
						'state': calc.state, 'extra': calc.extra, \
						'misc': calc.misc, 'mpi': calc.mpi, \
						'orbs': calc.orbs, 'restart': calc.restart}
				mpi.global_comm.bcast(info, root=0)
			else:
				info = mpi.global_comm.bcast(None, root=0)
				calc.model = info['model']; calc.target = info['target']
				calc.base = info['base']
				calc.thres = info['thres']; calc.prot = info['prot']
				calc.state = info['state']; calc.extra = info['extra']
				calc.misc = info['misc']; calc.mpi = info['mpi']
				calc.orbs = info['orbs']; calc.restart = info['restart']


def fund(mpi, mol, calc):
		""" bcast fundamental info """
		if mpi.parallel:
			if mpi.global_master:
				info = {'prop': calc.prop, \
							'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, \
							'ref_space': calc.ref_space, 'exp_space': calc.exp_space, \
							'occup': calc.occup, 'no_exp': calc.no_exp, \
							'ne_act': calc.ne_act, 'no_act': calc.no_act}
				mpi.global_comm.bcast(info, root=0)
				# bcast mo
				mpi.global_comm.Bcast([calc.mo, MPI.DOUBLE], root=0)
			else:
				info = mpi.global_comm.bcast(None, root=0)
				calc.prop = info['prop']
				mol.norb = info['norb']; mol.nocc = info['nocc']; mol.nvirt = info['nvirt']
				calc.ref_space = info['ref_space']; calc.exp_space = info['exp_space']
				calc.occup = info['occup']; calc.no_exp = info['no_exp']
				calc.ne_act = info['ne_act']; calc.no_act = info['no_act']
				# receive mo
				buff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
				mpi.global_comm.Bcast([buff, MPI.DOUBLE], root=0)
				calc.mo = buff
		if mol.atom:
			calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo)
		else:
			calc.orbsym = np.zeros(mol.norb, dtype=np.int)


def exp(mpi, calc, exp, comm):
		""" bcast exp info """
		if mpi.parallel:
			if mpi.global_master:
				# collect info
				info = {'len_tup': [len(exp.tuples[i]) for i in range(len(exp.tuples))], \
							'min_order': exp.min_order, 'start_order': exp.start_order}
				if calc.target['energy']:
					info['len_e_inc'] = [exp.prop['energy']['inc'][i].size for i in range(len(exp.prop['energy']['inc']))]
				if calc.target['excitation']:
					info['len_exc_inc'] = [exp.prop['excitation']['inc'][i].size for i in range(len(exp.prop['excitation']['inc']))]
				if calc.target['dipole']:
					info['len_dip_inc'] = [exp.prop['dipole']['inc'][i].shape[0] for i in range(len(exp.prop['dipole']['inc']))]
				if calc.target['trans']:
					info['len_trans_inc'] = [exp.prop['trans']['inc'][i].shape[0] for i in range(len(exp.prop['trans']['inc']))]
				# bcast info
				comm.bcast(info, root=0)
				# bcast tuples and hashes
				for i in range(1,len(exp.tuples)):
					comm.Bcast([exp.tuples[i], MPI.INT], root=0)
				for i in range(1,len(exp.hashes)):
					comm.Bcast([exp.hashes[i], MPI.INT], root=0)
				# bcast increments
				if calc.target['energy']:
					for i in range(len(exp.prop['energy']['inc'])):
						comm.Bcast([exp.prop['energy']['inc'][i], MPI.DOUBLE], root=0)
				if calc.target['excitation']:
					for i in range(len(exp.prop['excitation']['inc'])):
						comm.Bcast([exp.prop['excitation']['inc'][i], MPI.DOUBLE], root=0)
				if calc.target['dipole']:
					for i in range(len(exp.prop['dipole']['inc'])):
						comm.Bcast([exp.prop['dipole']['inc'][i], MPI.DOUBLE], root=0)
				if calc.target['trans']:
					for i in range(len(exp.prop['trans']['inc'])):
						comm.Bcast([exp.prop['trans']['inc'][i], MPI.DOUBLE], root=0)
			else:
				# receive info
				info = comm.bcast(None, root=0)
				# set min_order and start_order
				exp.min_order = info['min_order']
				exp.start_order = info['start_order']
				# receive tuples and hashes
				for i in range(1, len(info['len_tup'])):
					buff = np.empty([info['len_tup'][i], (exp.start_order-calc.no_exp)+i], dtype=np.int32)
					comm.Bcast([buff, MPI.INT], root=0)
					exp.tuples.append(buff)
				for i in range(1, len(info['len_tup'])):
					buff = np.empty(info['len_tup'][i], dtype=np.int64)
					comm.Bcast([buff, MPI.INT], root=0)
					exp.hashes.append(buff)
				# receive increments
				if calc.target['energy']:
					for i in range(len(info['len_e_inc'])):
						buff = np.zeros(info['len_e_inc'][i], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop['energy']['inc'].append(buff)
				if calc.target['excitation']:
					for i in range(len(info['len_exc_inc'])):
						buff = np.zeros(info['len_exc_inc'][i], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop['excitation']['inc'].append(buff)
				if calc.target['dipole']:
					for i in range(len(info['len_dip_inc'])):
						buff = np.zeros([info['len_dip_inc'][i], 3], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop['dipole']['inc'].append(buff)
				if calc.target['trans']:
					for i in range(len(info['len_trans_inc'])):
						buff = np.zeros([info['len_trans_inc'][i], 3], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop['trans']['inc'].append(buff)


def mbe(prop, comm):
		""" Allreduce property """
		comm.Allreduce(MPI.IN_PLACE, prop, op=MPI.SUM)


def screen(child_tup, child_hash, order, comm):
		""" Allgatherv tuples / hashes """
		# allgatherv tuples
		child_tup = np.asarray(child_tup, dtype=np.int32)
		recv_counts = np.array(comm.allgather(child_tup.size), dtype=np.int32)
		tuples = np.empty(np.sum(recv_counts, dtype=np.int64), dtype=np.int32)
		comm.Allgatherv(child_tup, [tuples, recv_counts])
		tuples = tuples.reshape(-1, order+1)
		if tuples.shape[0] > 0:
			# allgatherv hashes
			child_hash = np.asarray(child_hash, dtype=np.int64)
			recv_counts = np.array(comm.allgather(child_hash.size), dtype=np.int64)
			hashes = np.empty(np.sum(recv_counts, dtype=np.int64), dtype=np.int64)
			comm.Allgatherv(child_hash, [hashes, recv_counts])
			# sort wrt hashes
			tuples = tuples[hashes.argsort()]
			# sort hashes
			hashes.sort()
		else:
			hashes = np.array([], dtype=np.int64)
		return tuples, hashes


def final(mpi):
		""" terminate calculation """
		if mpi.global_master:
			restart.rm()
			if mpi.parallel:
				mpi.local_comm.bcast({'task': 'exit'}, root=0)
				if mpi.local_comm != mpi.global_comm:
					mpi.master_comm.bcast({'task': 'exit'}, root=0)
		elif mpi.local_master:
			mpi.local_comm.bcast({'task': 'exit'}, root=0)
		mpi.global_comm.Barrier()
		MPI.Finalize()


