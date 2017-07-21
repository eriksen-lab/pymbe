#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI

from bg_kernel import KernCls
from bg_ent import EntCls
from bg_screen import ScrCls


class DrvCls():
		""" driver class """
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" main driver routine """
				# make kernel, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.entanglement = EntCls()
				self.screening = ScrCls(_exp)
				# print expansion header
				_prt.exp_header()
				# now do expansion
				for _exp.order in range(_calc.exp_min_order,_calc.exp_max_order + 1):
					#
					#** energy kernel phase **#
					#
					# print kernel header
					_prt.kernel_header(_exp)
					# init e_inc
					if (len(_exp.energy_inc) != _exp.order):
						_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
					# kernel calculations
					self.kernel.main(_mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst)
					# print kernel end
					_prt.kernel_end(_calc, _exp)
					# print results
					_prt.kernel_results(_exp)
					#
					#** screening phase **#
					#
					# print screen header
					_prt.screen_header(_exp)
					# orbital entanglement
					self.entanglement.main(_mpi, _mol, _calc, _exp, _time, _rst)
					# orbital screening
					if ((not _exp.conv_energy[-1]) and (_exp.order < _calc.exp_max_order)):
						# perform screening
						self.screening.main(_mpi, _calc, _exp, _time, _rst)
						# write restart files
						if (not _exp.conv_orb[-1]):
							_rst.write_screen(_mpi, _exp, _time)
						# print screen results
						_prt.screen_results(_exp)
						# print screen end
						_prt.screen_end(_exp)
					else:
						# print screen end
						_prt.screen_end(_exp)
						break
					# update restart frequency
					_rst.rst_freq = _rst.update()
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return
	
	
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _rst):
				""" main slave routine """
				# make kernel, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.entanglement = EntCls()
				self.screening = ScrCls(_exp)
				# set loop/waiting logical
				slave = True
				# enter slave state
				while (slave):
					# task id
					msg = _mpi.comm.bcast(None, root=0)
					#
					#** energy kernel phase **#
					#
					if (msg['task'] == 'kernel_slave'):
						_exp.order = msg['order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, _exp, _time)
						_time.coll_kernel_time(_mpi, None, _exp.order)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'ent_abs_par'):
						_exp.order = msg['order']
						self.entanglement.ent_abs_par(_mpi, _exp, _time)
						_time.coll_screen_time(_mpi, None, _exp.order, msg['conv_energy'])
					elif (msg['task'] == 'screen_slave'):
						_exp.order = msg['order']
						_exp.thres = msg['thres']
						self.screening.slave(_mpi, _calc, _exp, _time)
						_time.coll_screen_time(_mpi, None, _exp.order, True)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				_mpi.final(None)
	
	
