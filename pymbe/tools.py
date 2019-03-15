#!/usr/bin/env python
# -*- coding: utf-8 -*

""" tools.py: various pure functions """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import os
import sys
import traceback
import subprocess
import numpy as np
import scipy.special
import math
from pyscf import lo

import parallel

# output folder and files
OUT = os.getcwd()+'/output'
OUT_FILE = OUT+'/output.out'
RES_FILE = OUT+'/results.out'
# array of degenerate (dooh) orbsym IDs
# E1gx (2) , E1gy (3)
# E1uy (6) , E1ux (7)
DEG_ID = np.array([2, 6]) 


class Logger(object):
		""" write to both stdout and output_file """
		def __init__(self, output_file, both=True):
			self.terminal = sys.stdout
			self.log = open(output_file, 'a')
			self.both = both
		def write(self, message):
			self.log.write(message)
			if self.both:
				self.terminal.write(message)
		def flush(self):
			pass


def enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


def fsum(a):
		""" use math.fsum to safely sum 1d array or 2d array (column-wise) """
		if a.ndim == 1:
			return math.fsum(a)
		elif a.ndim == 2:
			return np.fromiter(map(math.fsum, a.T), dtype=a.dtype, count=a.shape[1])
		else:
			raise NotImplementedError('tools.py: _fsum()')


def hash_2d(a):
		""" convert a 2d numpy array to a 1d array of hashes """
		return np.fromiter(map(hash_1d, a), dtype=np.int64, count=a.shape[0])


def hash_1d(a):
		""" convert a 1d numpy array to a hash """
		return hash(a.tobytes())


def hash_compare(a, b):
		""" find occurences of b in a from binary searches """
		left = a.searchsorted(b, side='left')
		right = a.searchsorted(b, side='right')
		if ((right - left) > 0).all():
			return left
		else:
			return None


def dict_conv(old_dict):
		""" convert dict keys """
		new_dict = {}
		for key, value in old_dict.items():
			if key.lower() in ['method', 'active', 'scheme']:
				new_dict[key.lower()] = value.lower()
			else:
				new_dict[key.lower()] = value
		return new_dict


def git_version():
		""" return the git revision as a string
		see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92
		"""
		def _minimal_ext_cmd(cmd):
			env = {}
			for k in ['SYSTEMROOT', 'PATH', 'HOME']:
				v = os.environ.get(k)
				if v is not None:
					env[k] = v
			# LANGUAGE is used on win32
			env['LANGUAGE'] = 'C'
			env['LANG'] = 'C'
			env['LC_ALL'] = 'C'
			out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
			return out
		try:
			out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
			GIT_REVISION = out.strip().decode('ascii')
		except OSError:
			GIT_REVISION = "Unknown"
		return GIT_REVISION


def get_pymbe_path():
		""" return path to pymbe """
		return os.path.dirname(os.path.realpath(sys.argv[0]))


def tasks(n_tuples, num_slaves, task_size):
		""" task list """
		if num_slaves * task_size < n_tuples:
			n_tasks = n_tuples // task_size
		else:
			n_tasks = num_slaves
		return np.array_split(np.arange(n_tuples), n_tasks)


def core_cas(mol, ref_space, tup):
		""" define core and cas spaces """
		cas_idx = cas(ref_space, tup)
		core_idx = np.setdiff1d(np.arange(mol.nocc), cas_idx)
		return core_idx, cas_idx


def cas(ref_space, tup):
		""" define cas space """
		return np.sort(np.append(ref_space, tup))


def sigma_prune(mo_energy, orbsym, tup, mbe=False):
		""" sigma pruning """
		# loop over IDs
		for sym in DEG_ID:
			# given set of x and y pi orbs
			pi_orbs = np.where((orbsym[tup] == sym) | (orbsym[tup] == (sym+1)))[0]
			if pi_orbs.size > 0:
				if pi_orbs.size % 2 > 0:
					# uneven number of pi orbs
					if mbe:
						# mbe phase
						return False
					if orbsym[tup[-1]] not in [sym, sym+1]:
						# last orbital is not a pi orbital
						return False
					else:
						if np.abs(mo_energy[tup[-1]] - mo_energy[tup[-1]-1]) < 1.0e-05:
							# this is the second member of a pair of degenerated pi orbs
							return False
				else:
					# even number of pi orbs
					for i in range(1, pi_orbs.size, 2):
						if np.abs(mo_energy[tup[pi_orbs[i]]] - mo_energy[tup[pi_orbs[i-1]]]) > 1.0e-05:
							# the pi orbs are not pair-wise degenerated
							return False
		return True


class hubbard_PM(lo.pipek.PM):
		""" Construct the site-population tensor for each orbital-pair density
			(pyscf example: 40-hubbard_model_PM_localization.py) """
		def atomic_pops(self, mol, mo_coeff, method=None):
			""" This tensor is used in cost-function and its gradients """
			return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)


def mat_indx(site, nx, ny):
		""" get x and y indices of a matrix """
		x = site % nx
		y = int(math.floor(float(site) / ny))
		return x, y


def near_nbrs(site_xy, nx, ny):
		""" get list of nearest neighbour indices """
		left = ((site_xy[0] - 1) % nx, site_xy[1])
		right = ((site_xy[0] + 1) % nx, site_xy[1])
		down = (site_xy[0], (site_xy[1] + 1) % ny)
		up = (site_xy[0], (site_xy[1] - 1) % ny)
		return [left, right, down, up]


def num_dets(n_orbs, n_alpha, n_beta):
		""" number of determinants in given CASCI calculation (ignoring spatial symmetry) """
		return scipy.special.binom(n_orbs, n_alpha) * scipy.special.binom(n_orbs, n_beta)


def assertion(condition, reason):
		""" assertion of condition """
		if not condition:
			# get stack
			stack = ''.join(traceback.format_stack()[:-1])
			# print stack
			print('\n\n'+stack)
			print('\n\n*** PyMBE assertion error: '+reason+' ***\n\n')
			# abort calculation
			parallel.abort()


