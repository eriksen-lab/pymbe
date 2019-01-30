#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from pyscf import symm
import sys
import itertools
import scipy.misc

import kernel
import output
import expansion
import driver
import parallel
import tools


# mbe parameters
TAGS = tools.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# init increments
		if calc.target['energy']:
			exp.prop['energy']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
		if calc.target['excitation']:
			exp.prop['excitation']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
		if calc.target['dipole']:
			exp.prop['dipole']['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
		if calc.target['trans']:
			exp.prop['trans']['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				# master
				_master(mpi, mol, calc, exp)
			else:
				# slaves
				_slave(mpi, mol, calc, exp)
				return
		else:
			_serial(mpi, mol, calc, exp)
		# sum up total quantities
		if mol.debug >= 1 and exp.order == 1:
			if calc.target['energy']:
				print('')
				a = np.abs(exp.prop['energy']['inc'][0])
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , root {1:} energy = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['excitation']:
				print('')
				a = np.abs(exp.prop['excitation']['inc'][0])
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , roots 0 -> {1:} excitation energy = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['dipole']:
				print('')
				a = np.linalg.norm(exp.prop['dipole']['inc'][0], axis=1)
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , root {1:} dipole = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['trans']:
				print('')
				a = np.linalg.norm(exp.prop['trans']['inc'][0], axis=1)
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , roots 0 -> {1:} trans = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																				calc.state['root'], a[np.argsort(a)[::-1]][j]))
			print('')
		if calc.target['energy']:
			exp.prop['energy']['tot'].append(tools.fsum(exp.prop['energy']['inc'][-1]))
		if calc.target['excitation']:
			exp.prop['excitation']['tot'].append(tools.fsum(exp.prop['excitation']['inc'][-1]))
		if calc.target['dipole']:
			exp.prop['dipole']['tot'].append(tools.fsum(exp.prop['dipole']['inc'][-1]))
		if calc.target['trans']:
			exp.prop['trans']['tot'].append(tools.fsum(exp.prop['trans']['inc'][-1]))
		if exp.order > exp.start_order:
			if calc.target['energy']:
				exp.prop['energy']['tot'][-1] += exp.prop['energy']['tot'][-2]
			if calc.target['excitation']:
				exp.prop['excitation']['tot'][-1] += exp.prop['excitation']['tot'][-2]
			if calc.target['dipole']:
				exp.prop['dipole']['tot'][-1] += exp.prop['dipole']['tot'][-2]
			if calc.target['trans']:
				exp.prop['trans']['tot'][-1] += exp.prop['trans']['tot'][-2]


def _serial(mpi, mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# init counter
		exp.count.append(0)
		# loop over tuples
		for i in range(len(exp.tuples[-1])):
			# calculate increments
			if not calc.extra['sigma'] or (calc.extra['sigma'] and tools.sigma_prune(calc.orbsym, exp.tuples[-1][i], mbe=True)):
				_calc(mpi, mol, calc, exp, i)
				exp.count[-1] += 1
				# print status
				output.mbe_status(exp, float(i+1) / float(len(exp.tuples[-1])))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _master(mpi, mol, calc, exp):
		""" master function """
		# set communicator
		comm = mpi.local_comm
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		comm.bcast(msg, root=0)
		# start time
		time = MPI.Wtime()
		# number of slaves
		num_slaves = slaves_avail = min(mpi.local_size - 1, len(exp.tuples[-1]))
		# task list and number of tasks
		tasks = tools.mbe_tasks(len(exp.tuples[-1]), num_slaves, calc.mpi['task_size'])
		n_tasks = len(tasks)
		# init request
		req = MPI.Request()
		# start index
		i = 0
		# loop until no tasks left
		while True:
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < n_tasks:
					# send index
					comm.Isend([np.array([i], dtype=np.int32), MPI.INT], dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
					# wait for completion
					req.Wait()
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for completion
					req.Wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# init counter
		exp.count.append(0)
		# allreduce properties
		if calc.target['energy']:
			parallel.mbe(exp.prop['energy']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(exp.prop['energy']['inc'][-1])
		if calc.target['excitation']:
			parallel.mbe(exp.prop['excitation']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(exp.prop['excitation']['inc'][-1])
		if calc.target['dipole']:
			parallel.mbe(exp.prop['dipole']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(np.count_nonzero(exp.prop['dipole']['inc'][-1], axis=1))
		if calc.target['trans']:
			parallel.mbe(exp.prop['trans']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(np.count_nonzero(exp.prop['trans']['inc'][-1], axis=1))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# set communicator
		comm = mpi.local_comm
		# init idx
		idx = np.empty(1, dtype=np.int32)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.local_size - 1, len(exp.tuples[-1]))
		# task list
		tasks = tools.mbe_tasks(len(exp.tuples[-1]), num_slaves, calc.mpi['task_size'])
		# send availability to master
		if mpi.local_rank <= num_slaves:
			comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.local_rank > num_slaves:
				break
			# receive index
			comm.Recv([idx, MPI.INT], source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				# get task
				task = tasks[idx[0]]
				# loop over tasks
				for n, task_idx in enumerate(task):
					# send availability to master
					if n == task.size - 1:
						comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					# calculate increments
					if not calc.extra['sigma'] or (calc.extra['sigma'] and tools.sigma_prune(calc.orbsym, exp.tuples[-1][task_idx], mbe=True)):
						_calc(mpi, mol, calc, exp, task_idx)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allreduce properties
		if calc.target['energy']:
			parallel.mbe(exp.prop['energy']['inc'][-1], comm)
		if calc.target['excitation']:
			parallel.mbe(exp.prop['excitation']['inc'][-1], comm)
		if calc.target['dipole']:
			parallel.mbe(exp.prop['dipole']['inc'][-1], comm)
		if calc.target['trans']:
			parallel.mbe(exp.prop['trans']['inc'][-1], comm)


def _calc(mpi, mol, calc, exp, idx):
		""" calculate increments """
		res = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
		if calc.target['energy']:
			exp.prop['energy']['inc'][-1][idx] = res['energy']
		if calc.target['excitation']:
			exp.prop['excitation']['inc'][-1][idx] = res['excitation']
		if calc.target['dipole']:
			exp.prop['dipole']['inc'][-1][idx] = res['dipole']
		if calc.target['trans']:
			exp.prop['trans']['inc'][-1][idx] = res['trans']


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = tools.core_cas(mol, calc.ref_space, tup)
		# perform calc
		res = kernel.main(mol, calc, exp, calc.model['method'])
		inc = {}
		if calc.target['energy']:
			inc['energy'] = res['energy']
			if calc.base['method'] is not None:
				res = kernel.main(mol, calc, exp, calc.base['method'])
				inc['energy'] -= res['energy']
			inc['energy'] -= calc.prop['ref']['energy']
		if calc.target['excitation']:
			inc['excitation'] = res['excitation']
			inc['excitation'] -= calc.prop['ref']['excitation']
		if calc.target['dipole']:
			if res['dipole'] is None:
				inc['dipole'] = np.zeros(3, dtype=np.float64)
			else:
				inc['dipole'] = res['dipole']
				inc['dipole'] -= calc.prop['ref']['dipole']
		if calc.target['trans']:
			if res['trans'] is None:
				inc['trans'] = np.zeros(3, dtype=np.float64)
			else:
				inc['trans'] = res['trans']
				inc['trans'] -= calc.prop['ref']['trans']
		if exp.order > exp.start_order:
			if calc.target['energy']:
				if inc['energy'] != 0.0:
					res = _sum(calc, exp, tup, 'energy')
					inc['energy'] -= res['energy']
			if calc.target['excitation']:
				if inc['excitation'] != 0.0:
					res = _sum(calc, exp, tup, 'excitation')
					inc['excitation'] -= res['excitation']
			if calc.target['dipole']:
				if np.any(inc['dipole'] != 0.0):
					res = _sum(calc, exp, tup, 'dipole')
					inc['dipole'] -= res['dipole']
			if calc.target['trans']:
				if np.any(inc['trans'] != 0.0):
					res = _sum(calc, exp, tup, 'trans')
					inc['trans'] -= res['trans']
		# debug print
		if mol.debug >= 1:
			tup_lst = [i for i in tup]
			tup_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[tup]]
			string = ' INC: order = {:} , tup = {:}\n'
			string += '      symmetry = {:}\n'
			form = (exp.order, tup_lst, tup_sym)
			if calc.target['energy']:
				string += '      correlation energy increment for state {:} = {:.4e}\n'
				form += (calc.state['root'], inc['energy'],)
			if calc.target['excitation']:
				string += '      excitation energy increment for root {:} = {:.4e}\n'
				form += (calc.state['root'], inc['excitation'],)
			if calc.target['dipole']:
				string += '      dipole moment increment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
				form += (calc.state['root'], *inc['dipole'],)
			if calc.target['trans']:
				string += '      transition dipole moment increment for excitation 0 -> {:} = ({:.4e}, {:.4e}, {:.4e})\n'
				form += (calc.state['root'], *inc['trans'],)
			print(string.format(*form))
		return inc


def _sum(calc, exp, tup, prop):
		""" recursive summation """
		# init res
		res = {}
		if prop == 'energy':
			res['energy'] = 0.0
		elif prop == 'excitation':
			res['excitation'] = 0.0
		elif prop == 'dipole':
			res['dipole'] = np.zeros(3, dtype=np.float64)
		elif prop == 'trans':
			res['trans'] = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for k in range(exp.order-exp.start_order, 0, -1):
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)
			# sigma pruning
			if calc.extra['sigma']:
				combs = combs[[tools.sigma_prune(calc.orbsym, combs[comb, :]) for comb in range(combs.shape[0])]]
			# convert to sorted hashes
			combs_hash = tools.hash_2d(combs)
			combs_hash.sort()
			# get indices
			indx = tools.hash_compare(exp.hashes[k-1], combs_hash)
			assert indx is not None
			# add up lower-order increments
			if prop == 'energy':
				res['energy'] += tools.fsum(exp.prop['energy']['inc'][k-1][indx])
			elif prop == 'excitation':
				res['excitation'] += tools.fsum(exp.prop['excitation']['inc'][k-1][indx])
			elif prop == 'dipole':
				res['dipole'] += tools.fsum(exp.prop['dipole']['inc'][k-1][indx, :])
			elif prop == 'trans':
				res['trans'] += tools.fsum(exp.prop['trans']['inc'][k-1][indx, :])
		return res




