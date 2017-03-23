#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_rst.py: mpi restart utilities for Bethe-Goldstone correlation calculations."""

from os import mkdir
from os.path import isdir

from bg_rst_read import read_rst

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def rst_init_env(molecule):
   #
   # base restart name
   #
   molecule['rst_name'] = 'rst'
   molecule['rst_dir'] = molecule['wrk_dir']+'/'+molecule['rst_name']
   #
   # sanity checks
   #
   if (molecule['rst'] and (not isdir(molecule['rst_dir']))):
      #
      print('restart requested but no rst directory present in work directory, aborting ...')
      #
      molecule['error'].append(True)
   #
   elif ((not molecule['rst']) and isdir(molecule['rst_dir'])):
      #
      print('no restart requested but rst directory present in work directory, aborting ...')
      #
      molecule['error'].append(True)  
   #
   # init main restart dir
   #
   if (not molecule['rst']): mkdir(molecule['rst_dir'])
   #
   return

def rst_main(molecule):
   #
   if (not molecule['rst']):
      #
      molecule['min_order'] = 1
   #
   else:
      #
      read_rst(molecule)
   #
   return molecule

