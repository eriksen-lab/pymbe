#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_print.py: general print utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from contextlib import redirect_stdout


class PrintCls():
		""" print functions """
		def __init__(self, _out):
				""" init parameters """
				self.out = _out.out_dir+'/bg_output.out'
				self.res = _out.out_dir+'/bg_results.out'
				# summary constants
				self.header_str = '{0:^93}'.format('-'*45)
				# print main header
				self.main_header()
				#
				return


		def main_header(self):
				""" print main header """
				for i in [self.out,self.res]:
					with open(i,'a') as f:
						with redirect_stdout(f):
							print('')
							print('')
							print("   oooooooooo.                .   oooo")
							print("   `888'   `Y8b             .o8   `888")
							print("    888     888  .ooooo.  .o888oo  888 .oo.    .ooooo.")
							print("    888oooo888' d88' `88b   888    888P'Y88b  d88' `88b")
							print("    888    `88b 888ooo888   888    888   888  888ooo888  888888")
							print("    888    .88P 888    .o   888 .  888   888  888    .o")
							print("   o888bood8P'  `Y8bod8P'   '888' o888o o888o `Y8bod8P'")
							print('')
							print("     .oooooo.              oooo        .o8               .")
							print("    d8P'  `Y8b             `888       '888             .o8")
							print("   888            .ooooo.   888   .oooo888   .oooo.o .o888oo  .ooooo.  ooo. .oo.    .ooooo.")
							print("   888           d88' `88b  888  d88' `888  d88(  '8   888   d88' `88b `888P'Y88b  d88' `88b")
							print("   888     ooooo 888   888  888  888   888  `'Y88b.    888   888   888  888   888  888ooo888")
							print("   `88.    .88'  888   888  888  888   888  o.  )88b   888 . 888   888  888   888  888    .o")
							print("    `Y8bood8P'   `Y8bod8P' o888o `Y8bod88P' `Y8888P'   '888' `Y8bod8P' o888o o888o `Y8bod8P'")
							print('')
							print('')
							print('   --- an incremental Python-based electronic structure correlation program written by:')
							print('')
							print('             Janus Juul Eriksen')
							print('')
							print('       with contributions from:')
							print('')
							print('             Filippo Lipparini')
							print('               & Juergen Gauss')
							print('')
							print('                                            *****')
							print('                                       ***************')
							print('                                            *****')
				#
				return


		def exp_header(self, _calc, _exp):
				""" print expansion header """
				if (_exp.level == 'macro'):
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print('\n\n'+self.header_str)
							print('{0:^93}'.format(_calc.exp_type+' expansion'))
							print(self.header_str+'\n\n')
					# write also to stdout
					print('\n\n'+self.header_str)
					print('{0:^93}'.format(_calc.exp_type+' expansion'))
					print(self.header_str+'\n\n')
				#
				return
		
		
		def kernel_header(self, _exp):
				""" print energy kernel header """
				if (_exp.level == 'macro'):
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print(' --------------------------------------------------------------------------------------------')
							print(' STATUS-MACRO: order = {0:>d} energy kernel started  ---  {1:d} tuples in total'.\
									format(_exp.order,len(_exp.tuples[-1])))
							print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-MACRO: order = {0:>d} energy kernel started  ---  {1:d} tuples in total'.\
							format(_exp.order,len(_exp.tuples[-1])))
					print(' --------------------------------------------------------------------------------------------')
				#
				return

		
		def kernel_status(self, _exp, _prog):
				""" print status bar """
				if (_exp.level == 'macro'):
					bar_length = 50
					status = ""
					block = int(round(bar_length * _prog))
					print(' STATUS-MACRO:   [{0}]   ---  {1:>6.2f} % {2}'.\
							format('#' * block + '-' * (bar_length - block), _prog * 100, status))
				#
				return
	
	
		def kernel_end(self, _calc, _exp):
				""" print end of kernel """
				if (_exp.level == 'macro'):
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							if (_exp.conv_energy[-1]):
								print(' --------------------------------------------------------------------------------------------')
								print(' STATUS-MACRO: order = {0:>d} kernel done (E = {1:.6e}, threshold = {2:<5.2e})'.\
										format(_exp.order,np.sum(_exp.energy_inc[-1]),_calc.energy_thres))
								print(' STATUS-MACRO:                  *** convergence has been reached ***                         ')
								print(' --------------------------------------------------------------------------------------------')
							else:
								print(' --------------------------------------------------------------------------------------------')
								print(' STATUS-MACRO: order = {0:>d} kernel done (E = {1:.6e}, thres. = {2:<5.2e})'.\
										format(_exp.order,np.sum(_exp.energy_inc[-1]),_calc.energy_thres))
								print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					if (_exp.conv_energy[-1]):
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-MACRO: order = {0:>d} kernel done (E = {1:.6e}, threshold = {2:<5.2e})'.\
								format(_exp.order,np.sum(_exp.energy_inc[-1]),_calc.energy_thres))
						print(' STATUS-MACRO:                  *** convergence has been reached ***                         ')
						print(' --------------------------------------------------------------------------------------------')
					else:
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-MACRO: order = {0:>d} kernel done (E = {1:.6e}, thres. = {2:<5.2e})'.\
								format(_exp.order,np.sum(_exp.energy_inc[-1]),_calc.energy_thres))
						print(' --------------------------------------------------------------------------------------------')
				#
				return


		def kernel_micro_results(self, _calc, _exp):	
				""" print micro result statistics """
				if ((_calc.exp_type == 'combined') and (_exp.level == 'macro')):
					# statistics
					mean_val = int(np.mean(_exp.micro_conv_res) + 0.5)
					min_val = _exp.micro_conv_res[np.argmin(_exp.micro_conv_res)]
					max_val = _exp.micro_conv_res[np.argmax(_exp.micro_conv_res)]
					if (len(_exp.micro_conv_res) > 1):
						std_val = np.std(_exp.micro_conv_res, ddof=1)
					else:
						std_val = 0.0
					# now print
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print(' --------------------------------------------------------------------------------------------')
							print(' RESULT-MICRO:     mean order    |      min. order     |      max. order     |    std.dev.   ')
							print(' --------------------------------------------------------------------------------------------')
							print(' RESULT-MICRO:   {0:>8d}        |    {1:>8d}         |    {2:>8d}         |   {3:<13.4e}'.\
									format(mean_val, min_val, max_val, std_val))
							print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					print(' --------------------------------------------------------------------------------------------')
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MICRO:     mean order    |      min. order     |      max. order     |    std.dev.   ')
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MICRO:   {0:>8d}        |    {1:>8d}         |    {2:>8d}         |   {3:<13.4e}'.\
							format(mean_val, min_val, max_val, std_val))
					print(' --------------------------------------------------------------------------------------------')
				#
				return


	
		def kernel_macro_results(self, _exp):
				""" print macro result statistics """
				if (_exp.level == 'macro'):
					# statistics
					mean_val = np.mean(_exp.energy_inc[-1])
					min_val = _exp.energy_inc[-1][np.argmin(np.abs(_exp.energy_inc[-1]))]
					max_val = _exp.energy_inc[-1][np.argmax(np.abs(_exp.energy_inc[-1]))]
					if (len(_exp.energy_inc[-1]) > 1):
						std_val = np.std(_exp.energy_inc[-1], ddof=1)
					else:
						std_val = 0.0
					# now print
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print(' --------------------------------------------------------------------------------------------')
							print(' RESULT-MACRO:     mean cont.    |   min. abs. cont.   |   max. abs. cont.   |    std.dev.   ')
							print(' --------------------------------------------------------------------------------------------')
							print(' RESULT-MACRO:  {0:>13.4e}    |  {1:>13.4e}      |  {2:>13.4e}      |   {3:<13.4e}'.\
									format(mean_val, min_val, max_val, std_val))
							print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MACRO:     mean cont.    |   min. abs. cont.   |   max. abs. cont.   |    std.dev.   ')
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MACRO:  {0:>13.4e}    |  {1:>13.4e}      |  {2:>13.4e}      |   {3:<13.4e}'.\
							format(mean_val, min_val, max_val, std_val))
					print(' --------------------------------------------------------------------------------------------')
				#
				return
		
		
		def screen_header(self, _exp):
				""" print screening header """
				if (_exp.level == 'macro'):
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print(' --------------------------------------------------------------------------------------------')
							print(' STATUS-MACRO: order = {0:>d} screening started'.format(_exp.order))
							print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-MACRO: order = {0:>d} screening started'.format(_exp.order))
					print(' --------------------------------------------------------------------------------------------')
				#
				return
		
		
		def screen_results(self, _exp):
				""" print screening results """
				if (_exp.level == 'macro'):
					if (len(_exp.tuples) > _exp.order):
						screen = (1.0 - (len(_exp.tuples[-1]) / \
									(len(_exp.tuples[-1]) + _exp.screen_count[-1]))) * 100.0
					else:
						screen = 100.0
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							print(' --------------------------------------------------------------------------------------------')
							print(' UPDATE-MACRO: threshold value of {0:.2e} resulted in screening of {1:.2f} % of the tuples'.\
									format(_exp.thres,screen))
							print(' --------------------------------------------------------------------------------------------')
					# write also to stdout
					print(' --------------------------------------------------------------------------------------------')
					print(' UPDATE-MACRO: threshold value of {0:.2e} resulted in screening of {1:.2f} % of the tuples'.\
							format(_exp.thres,screen))
					print(' --------------------------------------------------------------------------------------------')
				#
				return
		
		
		def screen_end(self, _exp):
				""" print end of screening """
				if (_exp.level == 'macro'):
					with open(self.out,'a') as f:
						with redirect_stdout(f):
							if (_exp.conv_orb[-1]):
								print(' --------------------------------------------------------------------------------------------')
								print(' STATUS-MACRO: order = {0:>d} screening done'.format(_exp.order))
								print(' STATUS-MACRO:                  *** convergence has been reached ***                         ')
								print(' --------------------------------------------------------------------------------------------\n\n')
							else:
								print(' --------------------------------------------------------------------------------------------')
								print(' STATUS-MACRO: order = {0:>d} screening done'.format(_exp.order))
								print(' --------------------------------------------------------------------------------------------\n\n')
					# write also to stdout
					if (_exp.conv_orb[-1]):
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-MACRO: order = {0:>d} screening done'.format(_exp.order))
						print(' STATUS-MACRO:                  *** convergence has been reached ***                         ')
						print(' --------------------------------------------------------------------------------------------\n\n')
					else:
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-MACRO: order = {0:>d} screening done'.format(_exp.order))
						print(' --------------------------------------------------------------------------------------------\n\n')
				#
				return
		
		
