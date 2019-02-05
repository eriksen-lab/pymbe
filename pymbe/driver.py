#!/usr/bin/env python
# -*- coding: utf-8 -*

""" driver.py: driver module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import restart
import mbe
import kernel
import output
import screen
import expansion
import parallel


def master(mpi, mol, calc, exp):
		""" master routine """
		# print expansion headers
		if mpi.master:
			output.main_header()
			output.exp_header(calc, exp)
		# restart
		if mpi.master and calc.restart:
			exp.thres, exp.rst_freq, calc.restart = _rst_print(mol, calc, exp)
		# now do expansion
		for exp.order in range(exp.min_order, exp.max_order+1):
			#** mbe phase **#
			if mpi.master:
				# print header
				output.mbe_header(exp)
			if len(exp.tuples) > len(exp.count):
				mbe.main(mpi, mol, calc, exp)
				if mpi.master:
					# write restart files
					restart.mbe_write(calc, exp)
			if mpi.master:
				# print mbe end
				output.mbe_end(calc, exp)
				# print mbe results
				output.mbe_results(mol, calc, exp)
			#** screening phase **#
			if exp.order < exp.max_order:
				# perform screening
				screen.main(mpi, mol, calc, exp)
				if mpi.master:
					# write restart files
					if exp.tuples[-1].shape[0] > 0: restart.screen_write(exp)
					# print screen end
					output.screen_end(exp)
			else:
				if mpi.master:
					# collect time
					exp.time['screen'].append(0.0)
			# update restart frequency
			if mpi.master: exp.rst_freq = max(exp.rst_freq // 2, 1)
			# convergence check
			if exp.tuples[-1].shape[0] == 0 or exp.order == exp.max_order:
				# timings
				exp.time['mbe'] = np.asarray(exp.time['mbe'])
				exp.time['screen'] = np.asarray(exp.time['screen'])
				exp.time['total'] = exp.time['mbe'] + exp.time['screen']
				break


def slave(mpi, mol, calc, exp):
		""" slave routine """
		# set loop/waiting logical
		slave = True
		# enter slave state
		while slave:
			# task id
			msg = mpi.comm.bcast(None, root=0)
			#** mbe phase **#
			if msg['task'] == 'mbe':
				exp.order = msg['order']
				mbe.main(mpi, mol, calc, exp)
			#** screening phase **#
			elif msg['task'] == 'screen':
				exp.order = msg['order']
				screen.main(mpi, mol, calc, exp)
			#** exit **#
			elif msg['task'] == 'exit':
				slave = False
		# finalize
		parallel.final(mpi)
	

def _rst_print(mol, calc, exp):
		""" print output in case of restart """
		# init rst_freq
		rst_freq = exp.rst_freq
		for exp.order in range(exp.start_order, exp.min_order):
			output.mbe_header(exp)
			output.mbe_end(calc, exp)
			output.mbe_results(mol, calc, exp)
			thres = screen.update(exp.order, calc.thres['init'], calc.thres['relax'])
			output.screen_header(exp, thres)
			output.screen_end(exp)
			rst_freq = max(rst_freq // 2, 1)
		return thres, rst_freq, False

	
