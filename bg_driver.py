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
from bg_screen import ScrCls
from bg_exp import ExpCls


class DrvCls():
		""" driver class """
		def __init__(self):
				""" init required classes """
				self.kernel = KernCls()
				self.screening = ScrCls()
				#
				return


		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" main driver routine """
				# exp class invocation on slaves
				if (_mpi.parallel and (_calc.exp_type in ['occupied','virtual'])):
					msg = {'task': 'exp_cls', 'type': _calc.exp_type, 'prim_tup': [], 'rst': _rst.restart}
					# bcast msg
					_mpi.comm.bcast(msg, root=0)
				# restart
				_rst.rst_main(_mpi, _calc, _exp, _time)
				# print expansion header
				_prt.exp_header(_calc, _exp)
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
					# write restart files
					_rst.write_kernel(_mpi, _exp, _time, True)
					# print results
					_prt.kernel_results(_exp)
					#
					#** screening phase **#
					#
					# print screen header
					_prt.screen_header(_exp)
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
					# update restart frequency
					_rst.rst_freq = _rst.update()
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return
	
	
		def slave(self, _mpi, _mol, _calc, _pyscf, _time):
				""" main slave routine """
				# set loop/waiting logical
				slave = True
				# enter slave state
				while (slave):
					# task id
					msg = _mpi.comm.bcast(None, root=0)
					#
					#** exp class invocation **#
					#
					if (msg['task'] == 'exp_cls'):
						exp = ExpCls(_mpi, _mol, _calc, msg['type'])
						# receive rst data
						if (msg['rst']): _mpi.bcast_rst(_calc, exp, _time)
					#
					#** energy kernel phase **#
					#
					if (msg['task'] == 'kernel_slave'):
						exp.order = msg['order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, exp, _time)
						_time.coll_phase_time(_mpi, None, exp.order, 'kernel')
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_slave'):
						exp.order = msg['order']
						exp.thres = msg['thres']
						self.screening.slave(_mpi, _calc, exp, _time)
						_time.coll_phase_time(_mpi, None, exp.order, 'screen')
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				_mpi.final(None)
	
	
