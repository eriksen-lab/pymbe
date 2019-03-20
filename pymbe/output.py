#!/usr/bin/env python
# -*- coding: utf-8 -*

""" output.py: print module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from pyscf import symm

import kernel
import tools


# output parameters
HEADER = '{0:^87}'.format('-'*45)
DIVIDER = ' '+'-'*92
FILL = ' '+'|'*92


def main_header():
		""" print main header """
		string = "\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
		string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
		string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888\n"
		string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8\n"
		string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"\n"
		string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
		string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
		string += "                .o..P'\n"
		string += "                `Y8P'\n\n\n"
		string += "   -- git version: {:}\n\n\n"
		form = (tools.git_version(),)
		return string.format(*form)


def exp_header(method):
		""" print expansion header """
		# set string
		string = HEADER+'\n'
		string += '{:^87}\n'
		string += HEADER+'\n\n'
		form = (method.upper()+' expansion',)
		return string.format(*form)


def mbe_header(n_tuples, n_orbs, order):
		""" print mbe header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:} MBE started  ---  {:} tuples in total (each w/ {:} MOs)\n'
		string += DIVIDER
		form = (order, n_tuples, n_orbs)
		return string.format(*form)


def mbe_debug(mol, calc, exp, tup, ndets_tup, nelec_tup, inc_tup, cas_idx):
		""" print mbe debug information """
		# tup and symmetry
		tup_lst = [i for i in tup]
		if mol.atom:
			tup_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[tup]]
		else:
			tup_sym = ['A'] * tup.size
		string = ' INC: order = {:} , tup = {:} , space = ({:}e,{:}o) , n_dets = {:.2e}\n'
		string += '      symmetry = {:}\n'
		form = (exp.order, tup_lst, nelec_tup[0] + nelec_tup[1], cas_idx.size, ndets_tup, tup_sym)
		if calc.target in ['energy', 'excitation']:
			string += '      increment for root {:} = {:.4e}\n'
			form += (calc.state['root'], inc_tup,)
		else:
			string += '      increment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
			form += (calc.state['root'], *inc_tup,)
		return string.format(*form)


def mbe_end(n_count, n_orbs, order):
		""" print end of mbe """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:} MBE done  ---  {:} tuples in total (each w/ {:} MOs)\n'
		string += DIVIDER
		form = (order, n_count, n_orbs)
		return string.format(*form)


def mbe_results(mol, calc, exp):
		""" print mbe result statistics """
		if calc.target in ['energy', 'excitation']:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			prop_inc = exp.prop[calc.target]['inc'][exp.order-1]
			# statistics
			if prop_inc.any():
				mean_val = np.mean(prop_inc[np.nonzero(prop_inc)])
				min_val = np.min(np.abs(prop_inc[np.nonzero(prop_inc)]))
				max_val = np.max(np.abs(prop_inc[np.nonzero(prop_inc)]))
			else:
				mean_val = min_val = max_val = 0.0
			# calculate total inc
			if exp.order == 1:
				tot_inc = prop_tot[exp.order-1]
			else:
				tot_inc = prop_tot[exp.order-1] - prop_tot[exp.order-2]
			# set header
			if calc.target == 'energy':
				header = 'energy for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			else:
				header = 'excitation energy for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			# set string
			string += DIVIDER+'\n'
			string += ' RESULT:{:^81}\n'
			string += DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
			string += DIVIDER+'\n'
			string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
			form = (header, mean_val, min_val, max_val)
			# statistics
			mean_ndets = np.mean(exp.ndets[exp.order-1])
			min_ndets = np.min(exp.ndets[exp.order-1])
			max_ndets = np.max(exp.ndets[exp.order-1])
			string += DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:   mean # determinants   |      min. # determinants     |     max. # determinants\n'
			string += DIVIDER+'\n'
			string += ' RESULT:         {:>8.2e}        |           {:>9.3e}          |          {:>9.3e}\n'
			cas_idx_max = tools.cas(calc.ref_space, exp.tuples[exp.order-1][np.argmax(exp.ndets[exp.order-1])])
			nelec_max = np.asarray((np.count_nonzero(calc.occup[cas_idx_max] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_max] > 1.)), dtype=np.int32)
			cas_idx_min = tools.cas(calc.ref_space, exp.tuples[exp.order-1][np.argmin(exp.ndets[exp.order-1])])
			nelec_min = np.asarray((np.count_nonzero(calc.occup[cas_idx_min] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_min] > 1.)), dtype=np.int32)
			string += ' RESULT:         --------        |           ({:>2.0f}e,{:>2.0f}o)          |          ({:>2.0f}e,{:>2.0f}o)\n'
			string += DIVIDER+'\n'
			form += (mean_ndets, min_ndets, max_ndets, \
						nelec_min[0] + nelec_min[1], cas_idx_min.size, \
						nelec_max[0] + nelec_max[1], cas_idx_max.size)
		else:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			# calculate total inc
			if exp.order == 1:
				tot_inc = np.linalg.norm(prop_tot[exp.order-1])
			else:
				tot_inc = np.linalg.norm(prop_tot[exp.order-1]) - np.linalg.norm(prop_tot[exp.order-2])
			# set header
			if calc.target == 'dipole':
				header = 'dipole moment for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			else:
				header = 'transition dipole for excitation 0 -> {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			# set string/form
			string += DIVIDER+'\n'
			string += ' RESULT:{:^81}\n'
			string += DIVIDER+'\n'
			string += DIVIDER
			form = (header,)
			# set components
			comp = ('x-component', 'y-component', 'z-component')
			# init result arrays
			mean_val = np.empty(3, dtype=np.float64)
			min_val = np.empty(3, dtype=np.float64)
			max_val = np.empty(3, dtype=np.float64)
			# loop over x, y, and z
			for k in range(3):
				prop_inc = exp.prop[calc.target]['inc'][exp.order-1][:, k]
				# statistics
				if prop_inc.any():
					mean_val[k] = np.mean(prop_inc[np.nonzero(prop_inc)])
					min_val[k] = np.min(np.abs(prop_inc[np.nonzero(prop_inc)]))
					max_val[k] = np.max(np.abs(prop_inc[np.nonzero(prop_inc)]))
				else:
					mean_val[k] = min_val[k] = max_val[k] = 0.0
				string += '\n RESULT:{:^81}\n'
				string += DIVIDER+'\n'
				string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
				string += DIVIDER+'\n'
				string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}'
				if k < 2:
					string += '\n'+DIVIDER
				form += (comp[k], mean_val[k], min_val[k], max_val[k],)
			# statistics
			mean_ndets = np.mean(exp.ndets[exp.order-1])
			min_ndets = np.min(exp.ndets[exp.order-1])
			max_ndets = np.max(exp.ndets[exp.order-1])
			string += '\n'+DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:   mean # determinants   |      min. # determinants     |     max. # determinants\n'
			string += DIVIDER+'\n'
			string += ' RESULT:         {:>8.2e}        |           {:>9.3e}          |          {:>9.3e}\n'
			cas_idx_max = tools.cas(calc.ref_space, exp.tuples[exp.order-1][np.argmax(exp.ndets[exp.order-1])])
			nelec_max = np.asarray((np.count_nonzero(calc.occup[cas_idx_max] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_max] > 1.)), dtype=np.int32)
			cas_idx_min = tools.cas(calc.ref_space, exp.tuples[exp.order-1][np.argmin(exp.ndets[exp.order-1])])
			nelec_min = np.asarray((np.count_nonzero(calc.occup[cas_idx_min] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_min] > 1.)), dtype=np.int32)
			string += ' RESULT:         --------        |           ({:>2.0f}e,{:>2.0f}o)          |          ({:>2.0f}e,{:>2.0f}o)\n'
			string += DIVIDER+'\n'
			form += (mean_ndets, min_ndets, max_ndets, \
						nelec_min[0] + nelec_min[1], cas_idx_min.size, \
						nelec_max[0] + nelec_max[1], cas_idx_max.size)
		if exp.order < exp.max_order:
			string += FILL
		else:
			string += '\n\n'
		return string.format(*form)


def screen_header(order):
		""" print screening header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:} screening started\n'
		string += DIVIDER
		form = (order,)
		return string.format(*form)


def screen_end(n_tuples, order):
		""" print end of screening """
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:} screening done\n'
		if n_tuples == 0:
			string += ' STATUS:                  *** convergence has been reached ***                         \n'
		string += DIVIDER+'\n\n'
		form = (order,)
		return string.format(*form)
		
	
