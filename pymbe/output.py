#!/usr/bin/env python
# -*- coding: utf-8 -*

""" output.py: print module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
import contextlib

import kernel


# output parameters
OUT = os.getcwd()+'/output'
HEADER = '{0:^87}'.format('-'*45)
DIVIDER = ' '+'-'*92
FILL = ' '+'|'*92


def main_header():
		""" print main header """
		# rm out if present
		if os.path.isdir(OUT): shutil.rmtree(OUT, ignore_errors=True)
		# mkdir out
		os.mkdir(OUT)
		# print headers
		for i in [OUT+'/output.out',OUT+'/results.out']:
			with open(i,'a') as f:
				with contextlib.redirect_stdout(f):
					print("\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo")
					print("   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8")
					print("    888   .d88' oooo    ooo  888b     d'888   888     888  888")
					print("    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8")
					print("    888           `88..8'    8  `888'   888   888    `88b  888    \"")
					print("    888            `888'     8    Y     888   888    .88P  888       o")
					print("   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8")
					print("                .o..P'")
					print("                `Y8P'\n\n")


def exp_header(calc, exp):
		""" print expansion header """
		# set string
		string = HEADER+'\n'
		string += '{0:^87}\n'
		string += HEADER+'\n\n'
		if calc.model['TYPE'] == 'OCC':
			form = ('occupied expansion')
		elif calc.model['TYPE'] == 'VIRT':
			form = ('virtual expansion')
		elif calc.model['TYPE'] == 'COMB':
			form = ('combined expansion')
		_print(string, form)


def mbe_header(exp):
		""" print mbe header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} MBE started  ---  {1:d} tuples in total\n'
		string += DIVIDER
		form = (exp.order, len(exp.tuples[exp.order-exp.start_order]))
		_print(string, form)


def mbe_status(exp, prog):
		""" print status bar """
		# write only to stdout
		bar_length = 50
		status = ""
		block = int(round(bar_length * prog))
		print(' STATUS:   [{0}]   ---  {1:>6.2f} % {2}'.\
				format('#' * block + '-' * (bar_length - block), prog * 100, status))


def mbe_end(calc, exp):
		""" print end of mbe """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:>d} MBE done\n'
		string += DIVIDER
		form = (exp.order,)
		_print(string, form)


def mbe_results(mol, calc, exp):
		""" print mbe result statistics """
		for i in ['ENERGY', 'DIPOLE']:
			if i == 'ENERGY':
				for j in range(calc.state['ROOT']+1):
					print(FILL)
					prop_inc = exp.property['energy'][j]['inc'][exp.order-exp.start_order]
					prop_tot = exp.property['energy'][j]['tot']
					# statistics
					mean_val = np.mean(prop_inc)
					min_idx = np.argmin(np.abs(prop_inc))
					min_val = prop_inc[min_idx]
					max_idx = np.argmax(np.abs(prop_inc))
					max_val = prop_inc[max_idx]
					# calculate total inc
					if len(prop_tot) == 1:
						tot_inc = prop_tot[0]
					else:
						tot_inc = prop_tot[-1] - prop_tot[-2]
					# set header
					if j == 0:
						header = 'ground state energy (total increment = {:.4e})'.format(tot_inc)
					else:
						header = 'excitation energy for root {:} (total increment = {:.4e})'.format(j, tot_inc)
					# set string
					string = DIVIDER+'\n'
					string += ' RESULT:{:^81}\n'
					string += DIVIDER+'\n'
					string += DIVIDER+'\n'
					string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
					string += DIVIDER+'\n'
					string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
					string += DIVIDER
					form = (header, mean_val, min_val, max_val)
					_print(string, form)
			elif i == 'DIPOLE':
				for j in range(calc.state['ROOT']+1):
					print(FILL)
					prop_tot = exp.property['dipole'][j]['tot']
					# calculate total inc
					if len(prop_tot) == 1:
						tot_inc = np.sqrt(np.sum(prop_tot[0]**2))
					else:
						tot_inc = np.sqrt(np.sum(prop_tot[-1]**2)) - np.sqrt(np.sum(prop_tot[-2]**2))
					# set header
					if j == 0:
						header = 'ground state dipole moment (total increment = {:.4e})'.format(tot_inc)
					else:
						header = 'excitation dipole for root {:} (total increment = {:.4e})'.format(j, tot_inc)
					# set string/form
					string = DIVIDER+'\n'
					string += ' RESULT:{:^81}\n'
					string += DIVIDER+'\n'
					string += DIVIDER
					form = (header,)
					# set components
					comp = ('x-component', 'y-component', 'z-component')
					# init result arrays
					mean_val = np.empty(3, dtype=np.float64)
					min_idx = np.empty(3, dtype=np.int)
					min_val = np.empty(3, dtype=np.float64)
					max_idx = np.empty(3, dtype=np.int)
					max_val = np.empty(3, dtype=np.float64)
					# loop over x, y, and z
					for k in range(3):
						prop_inc = exp.property['dipole'][j]['inc'][exp.order-exp.start_order][:, k]
						# statistics
						mean_val[k] = np.mean(prop_inc)
						min_idx[k] = np.argmin(np.abs(prop_inc))
						min_val[k] = prop_inc[min_idx[k]]
						max_idx[k] = np.argmax(np.abs(prop_inc))
						max_val[k] = prop_inc[max_idx[k]]
						string += '\n RESULT:{:^81}\n'
						string += DIVIDER+'\n'
						string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
						string += DIVIDER+'\n'
						string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
						string += DIVIDER
						form += (comp[k], mean_val[k], min_val[k], max_val[k],)
					_print(string, form)
		print(FILL)
		# closing if no screening follows
		if exp.order == exp.max_order:
			with open(OUT+'/output.out','a') as f:
				with contextlib.redirect_stdout(f):
					print('\n\n')
			# write also to stdout
			print('\n\n')


def screen_header(exp, thres):
		""" print screening header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening started (thres. = {1:5.2e})\n'
		string += DIVIDER
		form = (exp.order, thres)
		_print(string, form)


def screen_end(exp):
		""" print end of screening """
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening done\n'
		if exp.tuples[-1].shape[0] == 0:
			string += ' STATUS:                  *** convergence has been reached ***                         \n'
		string += DIVIDER+'\n\n'
		form = (exp.order,)
		_print(string, form)
		
	
def _print(string, form):
		""" print to output file and stdout """
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(*form))
		print(string.format(*form))

	
