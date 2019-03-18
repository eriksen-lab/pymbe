#!/usr/bin/env python
# -*- coding: utf-8 -*

""" expansion.py: expansion class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import copy
import numpy as np

import tools


class ExpCls(object):
		""" expansion class """
		def __init__(self, mpi, mol, calc):
				""" init parameters """
				# set expansion model dict
				self.model = copy.deepcopy(calc.model)
				# init prop dict
				self.prop = {}
				self.prop[calc.target] = {'inc': []}
				if mpi.master:
					self.prop[calc.target]['tot'] = []
				# set max_order
				if calc.misc['order'] is not None:
					self.max_order = min(calc.exp_space.size, calc.misc['order'])
				else:
					self.max_order = calc.exp_space.size
				# init timings, calculation counter, and ndets lists
				self.count = []
				self.ndets = []
				self.time = {'mbe': [], 'screen': []}
				# init order (pre-calc)
				self.order = 0
				# restart frequency
				self.rst_freq = 50000


def init_tup(mol, calc):
		""" init tuples and hashes """
		# tuples
		if calc.extra['sigma']:
			tuples = [np.array([[i] for i in calc.exp_space if tools.sigma_prune(calc.mo_energy, \
						calc.orbsym, np.asarray([i], dtype=np.int32))], dtype=np.int32)]
		else:
			tuples = [np.array([[i] for i in calc.exp_space], dtype=np.int32)]
		# hashes
		hashes = [tools.hash_2d(tuples[0])]
		# sort wrt hashes
		tuples[0] = tuples[0][hashes[0].argsort()]
		hashes[0].sort()
		return tuples, hashes


