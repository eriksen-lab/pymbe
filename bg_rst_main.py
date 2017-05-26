#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_main.py: main restart utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from os import mkdir
from os.path import isdir

from bg_rst_read import rst_read_main
from bg_mpi_rst import rst_dist_master


def rst_init_env(molecule):
		""" init restart environment """
		# base restart name
		molecule['rst_name'] = 'rst'
		molecule['rst_dir'] = molecule['wrk_dir']+'/'+molecule['rst_name']
		# sanity checks
		if (molecule['rst'] and (not isdir(molecule['rst_dir']))):
			molecule['error_msg'] = 'restart requested but no rst directory present in work directory'
			molecule['error_code'] = 0
			molecule['error'].append(True)
		elif ((not molecule['rst']) and isdir(molecule['rst_dir'])):
			molecule['error_msg'] = 'no restart requested but rst directory present in work directory'
			molecule['error_code'] = 0
			molecule['error'].append(True)  
		# init main restart dir
		if (not molecule['rst']): mkdir(molecule['rst_dir'])
		#
		return


def rst_main(molecule):
		""" main restart driver """
		if (not molecule['rst']):
			# set start order for expansion
			molecule['min_order'] = 1
		else:
			# read in restart files
			rst_read_main(molecule)
			# distribute data to slaves
			if (molecule['mpi_parallel']): rst_dist_master(molecule)
			# update threshold
			rst_update_thres_and_rst_freq(molecule)
		#
		return


def rst_update_thres_and_rst_freq(molecule):
		""" update expansion thres and restart freq according to start order """
		for i in range(1,molecule['min_order']):
			molecule['prim_exp_thres'] = (molecule['prim_exp_scaling'])**(i) * molecule['prim_exp_thres_init']
			molecule['rst_freq'] /= 2.
		#
		return


