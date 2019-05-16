#!/usr/bin/env python
# -*- coding: utf-8 -*

""" screen.py: screening module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import itertools

import parallel
import tools
import output


# tags
TAGS = tools.enum('ready', 'tup', 'exit')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# master and slave functions
		if mpi.master:
			# start time
			time = MPI.Wtime()
			# master function
			tuples, hashes = _master(mpi, mol, calc, exp)
			# collect time
			exp.time['screen'].append(MPI.Wtime() - time)
			# append tuples and hashes
			exp.tuples.append(tuples)
			exp.hashes.append(hashes)
		else:
			# slave function
			hashes = _slave(mpi, mol, calc, exp)
			# append hashes
			exp.hashes.append(hashes)


def _master(mpi, mol, calc, exp):
		""" master function """
		# print header
		print(output.screen_header(exp.order))
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# number of tuples
		n_tuples = exp.tuples[-1].shape[0]
		# number of available slaves
		slaves_avail = min(mpi.size - 1, n_tuples)
		# tasks
		tasks = tools.tasks(n_tuples, slaves_avail, calc.mpi['task_size'])
		# init child_tup list
		child_tup = []
		# potential seed of occupied tuples for vacuum reference spaces
		if exp.min_order > 1 and exp.order <= calc.exp_space['occ'].size:
			# generate array with all k order subsets of occupied expansion space
			tuples_occ = np.array([tup for tup in itertools.combinations(calc.exp_space['occ'], exp.order)], \
									dtype=np.int32)
			# loop over occupied tuples
			for tup in tuples_occ:
				orbs = _orbs(mol, calc, exp, tup)
				# loop over orbitals
				for orb in orbs:
					child_tup += tup.tolist() + [orb]
		# loop until no tasks left
		for task in tasks:
			# set tups
			tups = exp.tuples[-1][task]
			# probe for available slaves
			mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)
			# receive slave status
			mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)
			# send tups
			mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)
		# done with all tasks
		while slaves_avail > 0:
			# probe for available slaves
			mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)
			# receive slave status
			mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)
			# send exit signal
			mpi.comm.isend(None, dest=mpi.stat.source, tag=TAGS.exit)
			# remove slave
			slaves_avail -= 1
		# allgatherv tuples
		child_tup = np.array(child_tup, dtype=np.int32)
		return parallel.screen(mpi, child_tup, exp.order)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# number of tasks
		n_tasks = exp.hashes[-1].size
		# number of needed slaves
		slaves_needed = min(mpi.size - 1, n_tasks)
		# init child_tup list
		child_tup = []
		# send availability to master
		if mpi.rank <= slaves_needed:
			mpi.comm.isend(None, dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.rank > slaves_needed:
				break
			# probe for task
			mpi.comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.tup:
				# number of elements in tups
				n_elms = mpi.stat.Get_elements(MPI.INT)
				# init tups
				tups = np.empty([n_elms // exp.order, exp.order], dtype=np.int32)
				# receive tups
				mpi.comm.Recv([tups, MPI.INT], source=0, tag=TAGS.tup)
				# loop over tups
				for tup in tups:
					# spawn child tuples from parent tuples at order k-1
					orbs = _orbs(mol, calc, exp, tup)
					# loop over orbitals
					for orb in orbs:
						child_tup += tup.tolist() + [orb]
				# send availability to master
				mpi.comm.isend(None, dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				mpi.comm.irecv(None, source=0, tag=TAGS.exit)
				break
		# allgatherv tuples
		child_tup = np.array(child_tup, dtype=np.int32)
		return parallel.screen(mpi, child_tup, exp.order)


def _orbs(mol, calc, exp, tup):
		""" determine list of child tuple orbitals """
		# set expansion space
		if exp.min_order == 1:
			exp_space = calc.exp_space['tot'][tup[-1] < calc.exp_space['tot']] 
		elif exp.min_order == 2:
			exp_space = calc.exp_space['virt'][tup[-1] < calc.exp_space['virt']] 
		# at min_order, spawn all possible child tuples
		if exp.order == exp.min_order:
			return np.array([orb for orb in exp_space], dtype=np.int32)
		# generate array with all k-1 order subsets of particular tuple
		combs = np.array([comb for comb in itertools.combinations(tup, exp.order-1)], dtype=np.int32)
		# prune combinations that will not result in cas spaces
		# with a mix of occupied and virtual orbitals
		combs = np.array([comb for comb in combs if tools.cas_allow(calc.occup, calc.ref_space, comb)], \
							dtype=np.int32)
		if combs.size == 0:
			return np.array([orb for orb in exp_space], dtype=np.int32)
		else:
			# init return list
			lst = []
			# loop over orbitals of expansion space
			for orb in exp_space:
				# add orbital to combinations
				orb_column = np.empty(combs.shape[0], dtype=np.int32)
				orb_column[:] = orb
				combs_orb = np.concatenate((combs, orb_column[:, None]), axis=1)
				# convert to sorted hashes
				combs_orb_hash = tools.hash_2d(combs_orb)
				combs_orb_hash.sort()
				# get indices
				idx = tools.hash_compare(exp.hashes[-1], combs_orb_hash)
				# add orbital to lst
				if idx is not None:
					# compute thresholds
					thres = np.fromiter(map(functools.partial(_thres, \
										calc.occup, calc.ref_space, calc.thres, \
										calc.prot['scheme']), combs_orb), \
										dtype=np.float64, count=idx.size)
					if not _prot_screen(calc.prot['scheme'], calc.target, exp.prop, thres, idx):
						lst += [orb]
			return np.array(lst, dtype=np.int32)


def _prot_screen(scheme, target, prop, thres, idx):
		""" protocol check """
		# all tuples have zero correlation
		if np.sum(thres) == 0.0:
			return False
		# extract increments with non-zero thresholds
		inc = prop[target]['inc'][-1][idx]
		inc = inc[np.nonzero(thres)]
		# screening procedure
		if target in ['energy', 'excitation']:
			return _prot_scheme(scheme, thres[np.nonzero(thres)], inc)
		else:
			screen = True
			for dim in range(3):
				# (x,y,z) = (0,1,2)
				if np.sum(inc[:, dim]) != 0.0:
					screen = _prot_scheme(scheme, thres[np.nonzero(thres)], inc[:, dim])
				if not screen:
					break
			return screen


def _prot_scheme(scheme, thres, prop):
		""" screen according to chosen scheme """
		if scheme == 1:
			# are *any* increments below their given threshold
			return np.any(np.abs(prop) < thres)
		elif scheme > 1:
			# are *all* increments below their given threshold
			return np.all(np.abs(prop) < thres)


def _thres(occup, ref_space, thres, scheme, tup):
		""" set screening threshold for tup """
		# involved dimensions
		nocc = np.count_nonzero(occup[ref_space] > 0.0)
		nocc += np.count_nonzero(occup[tup] > 0.0)
		nvirt = np.count_nonzero(occup[ref_space] == 0.0)
		nvirt += np.count_nonzero(occup[tup] == 0.0)
		# init thres
		threshold = 0.0
		# possibly update thres
		if nocc > 0 and nvirt > 0:
			if scheme < 3:
				# schemes 1 & 2
				if nvirt >= 3:
					threshold = thres['init'] * thres['relax'] ** (nvirt - 3)
			else:
				# scheme 3
				if max(nocc, nvirt) >= 3:
					threshold = thres['init'] * thres['relax'] ** (max(nocc, nvirt) - 3)
		return threshold


