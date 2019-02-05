#!/usr/bin/env python
# -*- coding: utf-8 -*

""" restart.py: restart module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
import json
import os
import os.path
import shutil
import re
from pyscf import symm


# rst parameters
RST = os.getcwd()+'/rst'


def restart():
		""" restart logical """
		if not os.path.isdir(RST):
			os.mkdir(RST)
			return False
		else:
			return True


def rm():
		""" remove rst directory in case of successful calc """
		shutil.rmtree(RST)


def main(calc, exp):
		""" main restart driver """
		if not calc.restart:
			return exp.start_order
		else:
			# list filenames in files list
			files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
			# sort the list of files
			files.sort(key=_natural_keys)
			# loop over files
			for i in range(len(files)):
				# read tuples
				if 'tup' in files[i]:
					exp.tuples.append(np.load(os.path.join(RST, files[i])))
				# read hashes
				elif 'hash' in files[i]:
					exp.hashes.append(np.load(os.path.join(RST, files[i])))
				# read increments
				elif 'e_inc' in files[i]:
					exp.prop['energy']['inc'].append(np.load(os.path.join(RST, files[i])))
				elif 'exc_inc' in files[i]:
					exp.prop['excitation']['inc'].append(np.load(os.path.join(RST, files[i])))
				elif 'dip_inc' in files[i]:
					exp.prop['dipole']['inc'].append(np.load(os.path.join(RST, files[i])))
				elif 'trans_inc' in files[i]:
					exp.prop['trans']['inc'].append(np.load(os.path.join(RST, files[i])))
				# read total properties
				elif 'e_tot' in files[i]:
					exp.prop['energy']['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				elif 'exc_tot' in files[i]:
					exp.prop['excitation']['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				elif 'dip_tot' in files[i]:
					exp.prop['dipole']['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				elif 'trans_tot' in files[i]:
					exp.prop['trans']['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				# read counter
				elif 'counter' in files[i]:
					exp.count.append(np.load(os.path.join(RST, files[i])).tolist())
				# read timings
				elif 'time_mbe' in files[i]:
					exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
				elif 'time_screen' in files[i]:
					exp.time['screen'].append(np.load(os.path.join(RST, files[i])).tolist())
			return exp.tuples[-1].shape[1] + calc.no_exp


def write_fund(mol, calc):
		""" write fundamental info restart files """
		# write dimensions
		dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'no_exp': calc.no_exp, \
				'ne_act': calc.ne_act, 'no_act': calc.no_act}
		with open(os.path.join(RST, 'dims.rst'), 'w') as f:
			json.dump(dims, f)
		# write hf, reference, and base properties
		if calc.target['energy']:
			energies = {'hf': calc.prop['hf']['energy'], \
						'base': calc.prop['base']['energy'], \
						'ref': calc.prop['ref']['energy']}
			with open(os.path.join(RST, 'energies.rst'), 'w') as f:
				json.dump(energies, f)
		if calc.target['excitation']:
			excitations = {'ref': calc.prop['ref']['excitation']}
			with open(os.path.join(RST, 'excitations.rst'), 'w') as f:
				json.dump(excitations, f)
		if calc.target['dipole']:
			dipoles = {'hf': calc.prop['hf']['dipole'].tolist(), \
						'ref': calc.prop['ref']['dipole'].tolist()}
			with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
				json.dump(dipoles, f)
		if calc.target['trans']:
			transitions = {'ref': calc.prop['ref']['trans'].tolist()}
			with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
				json.dump(transitions, f)
		# write expansion spaces
		np.save(os.path.join(RST, 'ref_space'), calc.ref_space)
		np.save(os.path.join(RST, 'exp_space'), calc.exp_space)
		# occupation
		np.save(os.path.join(RST, 'occup'), calc.occup)
		# write orbital energies
		np.save(os.path.join(RST, 'mo_energy'), calc.mo_energy)
		# write orbital coefficients
		np.save(os.path.join(RST, 'mo_coeff'), calc.mo_coeff)


def read_fund(mol, calc):
		""" read fundamental info restart files """
		# init zero dict
		calc.zero = {}
		# list filenames in files list
		files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
		# sort the list of files
		files.sort(key=_natural_keys)
		# loop over files
		for i in range(len(files)):
			# read dimensions
			if 'dims' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					dims = json.load(f)
				mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']; calc.no_exp = dims['no_exp']
				mol.norb = mol.nocc + mol.nvirt
				calc.ne_act = dims['ne_act']; calc.no_act = dims['no_act']
			# read hf and base properties
			elif 'energies' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					energies = json.load(f)
				calc.prop['hf']['energy'] = energies['hf']
				calc.prop['base']['energy'] = energies['base'] 
				calc.prop['ref']['energy'] = energies['ref']
			elif 'excitations' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					excitations = json.load(f)
				calc.prop['ref']['excitation'] = excitations['ref']
			elif 'dipoles' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					dipoles = json.load(f)
				calc.prop['hf']['dipole'] = np.asarray(dipoles['hf'])
				calc.prop['ref']['dipole'] = np.asarray(dipoles['ref'])
			elif 'transitions' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					transitions = json.load(f)
				calc.prop['ref']['trans'] = np.asarray(transitions['ref'])
			# read expansion spaces
			elif 'ref_space' in files[i]:
				calc.ref_space = np.load(os.path.join(RST, files[i]))
			elif 'exp_space' in files[i]:
				calc.exp_space = np.load(os.path.join(RST, files[i]))
			# read occupation
			elif 'occup' in files[i]:
				calc.occup = np.load(os.path.join(RST, files[i]))
			# read orbital energies
			elif 'mo_energy' in files[i]:
				calc.mo_energy = np.load(os.path.join(RST, files[i]))
			# read orbital coefficients
			elif 'mo_coeff' in files[i]:
				calc.mo_coeff = np.load(os.path.join(RST, files[i]))
				if mol.atom:
					calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
				else:
					calc.orbsym = np.zeros(mol.norb, dtype=np.int)


def mbe_write(calc, exp):
		""" write mbe restart files """
		# write incremental and total quantities
		# increments
		if calc.target['energy']:
			np.save(os.path.join(RST, 'e_inc_{:}'.format(exp.order)), exp.prop['energy']['inc'][-1])
		if calc.target['excitation']:
			np.save(os.path.join(RST, 'exc_inc_{:}'.format(exp.order)), exp.prop['excitation']['inc'][-1])
		if calc.target['dipole']:
			np.save(os.path.join(RST, 'dip_inc_{:}'.format(exp.order)), exp.prop['dipole']['inc'][-1])
		if calc.target['trans']:
			np.save(os.path.join(RST, 'trans_inc_{:}'.format(exp.order)), exp.prop['trans']['inc'][-1])
		# total properties
		if calc.target['energy']:
			np.save(os.path.join(RST, 'e_tot_{:}'.format(exp.order)), exp.prop['energy']['tot'][-1])
		if calc.target['excitation']:
			np.save(os.path.join(RST, 'exc_tot_{:}'.format(exp.order)), exp.prop['excitation']['tot'][-1])
		if calc.target['dipole']:
			np.save(os.path.join(RST, 'dip_tot_{:}'.format(exp.order)), exp.prop['dipole']['tot'][-1])
		if calc.target['trans']:
			np.save(os.path.join(RST, 'trans_tot_{:}'.format(exp.order)), exp.prop['trans']['tot'][-1])
		# write counter
		np.save(os.path.join(RST, 'counter_'+str(exp.order)), np.asarray(exp.count[-1]))
		# write time
		np.save(os.path.join(RST, 'time_mbe_'+str(exp.order)), np.asarray(exp.time['mbe'][-1]))


def screen_write(exp):
		""" write screening restart files """
		# write tuples
		np.save(os.path.join(RST, 'tup_'+str(exp.order+1)), exp.tuples[-1])
		# write hashes
		np.save(os.path.join(RST, 'hash_'+str(exp.order+1)), exp.hashes[-1])
		# write time
		np.save(os.path.join(RST, 'time_screen_'+str(exp.order)), np.asarray(exp.time['screen'][-1]))


def _natural_keys(txt):
		"""
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		cf. https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
		"""
		return [_convert(c) for c in re.split('(\d+)', txt)]


def _convert(txt):
		""" convert strings with numbers in them """
		return int(txt) if txt.isdigit() else txt


