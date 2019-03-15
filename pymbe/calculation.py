#!/usr/bin/env python
# -*- coding: utf-8 -*

""" calculation.py: calculation class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import re
import sys
import os
import ast
import numpy as np
from pyscf import symm

import tools
import restart


# components
COMP = ['model', 'base', 'orbs', 'target', 'prot', 'thres', 'mpi', 'extra', 'misc', 'ref', 'state']


class CalcCls(object):
		""" calculation class """
		def __init__(self, mpi, mol):
				""" init parameters """
				# set defaults
				self.model = {'method': 'fci', 'solver': 'pyscf_spin0'}
				self.target = {'energy': False, 'excitation': False, 'dipole': False, 'trans': False}
				self.prot = {'scheme': 'new'}
				self.ref = {'method': 'casci', 'hf_guess': True, 'active': 'manual', \
							'select': [i for i in range(mol.ncore, mol.nelectron // 2)], \
							'wfnsym': [symm.addons.irrep_id2name(mol.symmetry, 0) if mol.symmetry else 0]}
				self.base = {'method': None}
				self.state = {'wfnsym': symm.addons.irrep_id2name(mol.symmetry, 0) if mol.symmetry else 0, 'root': 0}
				self.extra = {'hf_guess': True, 'sigma': False}
				self.thres = {'init': 1.0e-10, 'relax': 1.0}
				self.misc = {'mem': 2000, 'order': None, 'async': False}
				self.orbs = {'type': 'canonical'}
				self.mpi = {'masters': 1, 'task_size': 1}
				# init mo
				self.mo = None
				# set calculation parameters
				if mpi.master:
					# read parameters
					self.model, self.target, self.prot, self.ref, \
						self.base, self.thres, self.state, self.extra, \
						self.misc, self.orbs, self.mpi = self.set_calc()
					# sanity check
					self.sanity_chk(mpi, mol)
					# set target
					self.target = [x for x in self.target.keys() if self.target[x]][0]
					# restart logical
					self.restart = restart.restart()
				# init prop dict
				self.prop = {'hf': {}, 'base': {}, 'ref': {}}


		def set_calc(self):
				""" set calculation and mpi parameters from input file """
				# read input file
				try:
					with open(os.getcwd()+'/input') as f:
						content = f.readlines()
						for i in range(len(content)):
							if content[i].strip():
								if content[i].split()[0][0] == '#':
									continue
								else:
									entry = re.split('=',content[i])[0].strip()
									if entry in COMP:
										try:
											tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
										except ValueError:
											raise ValueError('wrong input -- error in reading in {:} dictionary'.format(entry))
										tmp = tools.dict_conv(tmp)
										for key, val in tmp.items():
											if entry == 'model':
												self.model[key] = val
											elif entry == 'base':
												self.base[key] = val
											elif entry == 'orbs':
												self.orbs[key] = val
											elif entry == 'target':
												self.target[key] = val
											elif entry == 'prot':
												self.prot[key] = val
											elif entry == 'thres':
												self.thres[key] = val
											elif entry == 'mpi':
												self.mpi[key] = val
											elif entry == 'extra':
												self.extra[key] = val
											elif entry == 'misc':
												self.misc[key] = val
											elif entry == 'ref':
												if key == 'wfnsym':
													if not isinstance(val, list):
														self.ref[key] = list(val)
													self.ref[key] = [symm.addons.std_symb(self.ref[key][j]) for j in range(len(self.ref[key]))]
												else:
													self.ref[key] = val
											elif entry == 'state':
												if key == 'wfnsym':
													self.state[key] = symm.addons.std_symb(val)
												else:
													self.state[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				return self.model, self.target, self.prot, self.ref, self.base, \
							self.thres, self.state, self.extra, self.misc, self.orbs, self.mpi


		def sanity_chk(self, mpi, mol):
				""" sanity check for calculation and mpi parameters """
				# expansion model
				tools.assertion(isinstance(self.model['method'], str), \
								'input electronic structure method (method) must be a string')
				tools.assertion(self.model['method'] in ['ccsd', 'ccsd(t)', 'fci'], \
								'valid expansion models are: ccsd, ccsd(t), and fci')
				tools.assertion(self.model['solver'] in ['pyscf_spin0', 'pyscf_spin1'], \
								'valid FCI solvers are: pyscf_spin0 and pyscf_spin1')
				if self.model['method'] != 'fci':
					tools.assertion(self.model['solver'] == 'pyscf_spin0', \
									'setting a FCI solver for a non-FCI expansion model is not meaningful')
				if mol.spin > 0:
					tools.assertion(self.model['solver'] != 'pyscf_spin0', \
									'the pyscf_spin0 FCI solver is designed for spin singlets only')
				# reference model
				tools.assertion(self.ref['method'] in ['casci', 'casscf'], \
								'valid reference models are: casci and casscf')
				if self.ref['method'] == 'casscf':
					tools.assertion(self.model['method'] == 'fci', \
									'a casscf reference is only meaningful for an fci expansion model')
				tools.assertion(self.ref['active'] == 'manual', \
								'active space choices are currently: manual')
				tools.assertion(isinstance(self.ref['select'], list), \
								'select key (select) for active space must be a list of orbitals')
				tools.assertion(isinstance(self.ref['hf_guess'], bool), \
								'HF initial guess for CASSCF calc (hf_guess) must be a bool')
				if mol.atom:
					if self.ref['hf_guess']:
						tools.assertion(len(set(self.ref['wfnsym'])) == 1, \
										'illegal choice of ref wfnsym when enforcing hf initial guess')
						tools.assertion(self.ref['wfnsym'][0] == symm.addons.irrep_id2name(mol.symmetry, 0), \
										'illegal choice of ref wfnsym when enforcing hf initial guess')
					for i in range(len(self.ref['wfnsym'])):
						try:
							self.ref['wfnsym'][i] = symm.addons.irrep_name2id(mol.symmetry, self.ref['wfnsym'][i])
						except Exception as err:
							raise ValueError('illegal choice of ref wfnsym -- PySCF error: {:}'.format(err))
				# base model
				if self.base['method'] is not None:
					tools.assertion(self.ref['method'] == 'casci', \
									'use of base model is only permitted for casci expansion references')
					tools.assertion(self.target['energy'], \
									'use of base model is only permitted for target energies')
					tools.assertion(self.base['method'] in ['ccsd', 'ccsd(t)'], \
									'valid base models are currently: ccsd, and ccsd(t)')
				# state
				if mol.atom:
					try:
						self.state['wfnsym'] = symm.addons.irrep_name2id(mol.symmetry, self.state['wfnsym'])
					except Exception as err:
						raise ValueError('illegal choice of state wfnsym -- PySCF error: {:}'.format(err))
					tools.assertion(self.state['root'] >= 0, \
									'choice of target state (root) must be an int >= 0')
					if self.model['method'] != 'fci':
						tools.assertion(self.state['wfnsym'] == 0, \
										'illegal choice of wfnsym for chosen expansion model')
						tools.assertion(self.state['root'] == 0, \
										'excited states not implemented for chosen expansion model')
				# targets
				tools.assertion(any(self.target.values()) and len([x for x in self.target.keys() if self.target[x]]) == 1, \
								'one and only one target property must be requested')
				tools.assertion(all(isinstance(i, bool) for i in self.target.values()), \
								'values in target input (target) must be bools')
				tools.assertion(set(list(self.target.keys())) <= set(['energy', 'excitation', 'dipole', 'trans']), \
								'invalid choice for target property. valid choices are: '
								'energy, excitation energy (excitation), dipole, and transition dipole (trans)')
				if self.target['excitation']:
					tools.assertion(self.state['root'] > 0, \
									'calculation of excitation energy (excitation) requires target state root >= 1')
				if self.target['trans']:
					tools.assertion(self.target['excitation'], \
									'calculation of transition dipole moment (trans) '
									'requires calculation of excitation energy (excitation)')
				# extra
				tools.assertion(isinstance(self.extra['hf_guess'], bool), \
								'HF initial guess for FCI calcs (hf_guess) must be a bool')
				tools.assertion(isinstance(self.extra['sigma'], bool), \
								'sigma state pruning for FCI calcs (sigma) must be a bool')
				# screening protocol
				tools.assertion(all(isinstance(i, str) for i in self.prot.values()), \
								'values in prot input (prot) must be string and bools')
				tools.assertion(self.prot['scheme'] in ['new', 'old'], \
								'valid protocol schemes are: new and old')
				# expansion thresholds
				tools.assertion(all(isinstance(i, float) for i in self.thres.values()), \
								'values in threshold input (thres) must be floats')
				tools.assertion(set(list(self.thres.keys())) <= set(['init', 'relax']), \
								'valid input in thres dict is: init and relax')
				tools.assertion(self.thres['init'] >= 0.0, \
								'initial threshold (init) must be a float >= 0.0')
				tools.assertion(self.thres['relax'] >= 1.0, \
								'threshold relaxation (relax) must be a float >= 1.0')
				# orbital representation
				tools.assertion(self.orbs['type'] in ['canonical', 'dynamic', 'local', 'ccsd', 'ccsd(t)'], \
								'valid occupied orbital representations (occ) are currently: '
								'canonical (canonical), dynamic (dynamic), pipek-mezey (local), '
								'or natural orbs (ccsd or ccsd(t))')
				if self.orbs['type'] not in ['canonical', 'dynamic']:
					tools.assertion(self.ref['method'] == 'casci', \
									'static non-canonical orbitals requires casci expansion reference')
				if mol.atom and self.orbs['type'] == 'local':
					tools.assertion(mol.symmetry == 'C1', \
									'the combination of local orbs and point group symmetry '
									'different from c1 is not allowed')
				# misc
				tools.assertion(isinstance(self.misc['mem'], int) and self.misc['mem'] >= 1, \
								'maximum memory (mem) in units of MB must be an int >= 1')
				tools.assertion(isinstance(self.misc['order'], (int, type(None))), \
								'maximum expansion order (order) must be an int >= 1')
				if self.misc['order'] is not None:
					tools.assertion(self.misc['order'] >= 0, \
									'maximum expansion order (order) must be an int >= 1')
				tools.assertion(isinstance(self.misc['async'], bool), \
								'asynchronous key (async) must be a bool')
				# mpi
				tools.assertion(isinstance(self.mpi['task_size'], int) and self.mpi['task_size'] >= 1, \
								'size of mpi tasks (task_size) must be an int >= 1')


