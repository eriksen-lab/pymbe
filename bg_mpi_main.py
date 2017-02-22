#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_main.py: main MPI driver routine for Bethe-Goldstone correlation calculations."""

import numpy as np
from os import getcwd, mkdir, chdir
from shutil import copy, rmtree
from mpi4py import MPI

from bg_mpi_misc import print_mpi_table, mono_exp_merge_info
from bg_mpi_time import init_mpi_timings, collect_mpi_timings
from bg_mpi_energy import energy_kernel_mono_exp_par, energy_summation_par
from bg_mpi_orbitals import orb_generator_slave 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_mpi(molecule):
   #
   #  ---  master and slave routine
   #
   if (MPI.COMM_WORLD.Get_size() > 1):
      #
      molecule['mpi_parallel'] = True
   #
   else:
      #
      molecule['mpi_parallel'] = False
   #
   # slave proceed to the main slave routine
   #
   if (MPI.COMM_WORLD.Get_rank() != 0):
      #
      main_slave_rout(molecule)
   #
   else:
      #
      molecule['mpi_master'] = True
   #
   return molecule

def main_slave_rout(molecule):
   #
   #  ---  slave routine
   #
   slave = True
   #
   while (slave):
      #
      # start time
      #
      start_idle = MPI.Wtime()
      #
      msg = MPI.COMM_WORLD.bcast(None,root=0)
      #
      # bcast_mol_dict
      #
      if (msg['task'] == 'bcast_mol_dict'):
         #
         # collect idle time
         #
         end_idle = MPI.Wtime()-start_idle
         #
         # receive molecule dict from master
         #
         start_comm = MPI.Wtime()
         #
         molecule = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # set current mpi proc to 'slave'
         #
         molecule['mpi_master'] = False
         #
         # init slave mpi timings
         #
         init_mpi_timings(molecule)
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += end_idle
         #
         # collect mpi_time_comm_slave
         #
         molecule['mpi_time_comm_slave'] += MPI.Wtime()-start_comm
         #
         # overwrite wrk_dir in case this is different from the one on the master node
         #
         start_work = MPI.Wtime()
         #
         molecule['wrk'] = getcwd()
         #
         # update with private mpi info
         #
         molecule['mpi_comm'] = MPI.COMM_WORLD
         molecule['mpi_size'] = molecule['mpi_comm'].Get_size()
         molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
         molecule['mpi_name'] = MPI.Get_processor_name()
         molecule['mpi_stat'] = MPI.Status()
         #
         # collect mpi_time_work_slave
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
      #
      # init_slave_env
      #
      elif (msg['task'] == 'init_slave_env'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         start_work = MPI.Wtime()
         #
         # private scr dir
         #
         molecule['scr'] = molecule['wrk']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
         #
         # init scr env
         #
         mkdir(molecule['scr'])
         #
         chdir(molecule['scr'])
         #
         # init tuple lists
         #
         molecule['prim_tuple'] = []
         molecule['corr_tuple'] = []
         #
         # init e_inc lists
         #
         molecule['prim_energy_inc'] = []
         molecule['corr_energy_inc'] = []
         #
         # collect mpi_time_work_slave
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
      #
      # print_mpi_table
      #
      elif (msg['task'] == 'print_mpi_table'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         start_comm = MPI.Wtime()
         #
         print_mpi_table(molecule)
         #
         # collect mpi_time_comm_slave
         #
         molecule['mpi_time_comm_slave'] += MPI.Wtime()-start_comm
      #
      # mono_exp_merge_info
      #
      elif (msg['task'] == 'mono_exp_merge_info'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         molecule['min_corr_order'] = msg['min_corr_order']
         #
         mono_exp_merge_info(molecule)
      #
      # bcast_tuples
      #
      elif (msg['task'] == 'bcast_tuples'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         start_idle = MPI.Wtime()
         #
         final_msg = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # collect mpi_time_idle_init
         #
         molecule['mpi_time_idle_init'] += MPI.Wtime()-start_idle
         #
         start_comm = MPI.Wtime()
         #
         # receive the total number of tuples
         #
         tup_info = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # collect mpi_time_comm_init
         #
         molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
         #
         start_work = MPI.Wtime()
         #
         # init tup[k-1]
         #
         molecule['prim_tuple'].append(np.empty([tup_info['tot_tup'],msg['order']],dtype=np.int))
         #
         # collect mpi_time_work_init
         #
         molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
         #
         # receive the tuples
         #
         start_comm = MPI.Wtime()
         #
         MPI.COMM_WORLD.Bcast([molecule['prim_tuple'][-1],MPI.INT],root=0)
         #
         # collect mpi_time_comm_init
         #
         molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
         #
         tup_info.clear()
         final_msg.clear()
      #
      # orb_generator_par
      #
      elif (msg['task'] == 'orb_generator_par'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         # receive domain information
         #
         start_comm = MPI.Wtime()
         #
         dom_info = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # collect mpi_time_comm_init
         #
         molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
         #
         if (msg['level'] == 'MACRO'):
            #
            orb_generator_slave(molecule,dom_info['dom'],molecule['prim_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
            #
            start_idle = MPI.Wtime()
            #
            final_msg = MPI.COMM_WORLD.bcast(None,root=0)
            #
            # collect mpi_time_idle_init
            #
            molecule['mpi_time_idle_init'] += MPI.Wtime()-start_idle
            #   
            # receive the total number of tuples
            #
            start_comm = MPI.Wtime()
            #
            tup_info = MPI.COMM_WORLD.bcast(None,root=0)
            #
            # collect mpi_time_comm_init
            #
            molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
            #
            # init tup[k-1]
            #
            start_work = MPI.Wtime()
            #
            molecule['prim_tuple'].append(np.empty([tup_info['tot_tup'],msg['order']],dtype=np.int))
            #
            # collect mpi_time_work_init
            #
            molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
            #
            # receive the tuples
            #
            start_comm = MPI.Wtime()
            #
            MPI.COMM_WORLD.Bcast([molecule['prim_tuple'][-1],MPI.INT],root=0)
            #
            # collect mpi_time_comm_init
            #
            molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
         #
         elif (msg['level'] == 'CORRE'):
            #
            orb_generator_slave(molecule,dom_info['dom'],molecule['corr_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],'CORRE')
            #
            start_idle = MPI.Wtime()
            #
            final_msg = MPI.COMM_WORLD.bcast(None,root=0)
            #
            # collect mpi_time_idle_init
            #
            molecule['mpi_time_idle_init'] += MPI.Wtime()-start_idle
            #
            # receive the total number of tuples
            #
            start_comm = MPI.Wtime()
            #
            tup_info = MPI.COMM_WORLD.bcast(None,root=0)
            #
            # collect mpi_time_comm_init
            #
            molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
            #
            # init tup[k-1]
            #
            start_work = MPI.Wtime()
            #
            molecule['corr_tuple'].append(np.empty([tup_info['tot_tup'],msg['order']],dtype=np.int))
            #
            # collect mpi_time_work_init
            #
            molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
            #
            # receive the tuples
            #
            start_comm = MPI.Wtime()
            #
            MPI.COMM_WORLD.Bcast([molecule['corr_tuple'][-1],MPI.INT],root=0)
            #
            # collect mpi_time_comm_init
            #
            molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
         #
         dom_info.clear()
         tup_info.clear()
         final_msg.clear()
      #
      # energy_kernel_mono_exp_par
      #
      elif (msg['task'] == 'energy_kernel_mono_exp_par'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         if (msg['level'] == 'MACRO'):
            #
            energy_kernel_mono_exp_par(molecule,msg['order'],molecule['prim_tuple'],None,molecule['prim_energy_inc'],msg['l_limit'],msg['u_limit'],'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            energy_kernel_mono_exp_par(molecule,msg['order'],molecule['corr_tuple'],None,molecule['corr_energy_inc'],msg['l_limit'],msg['u_limit'],'CORRE')
      #
      # energy_summation_par
      #
      elif (msg['task'] == 'energy_summation_par'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         if (msg['level'] == 'MACRO'):
            #
            energy_summation_par(molecule,msg['order'],molecule['prim_tuple'],molecule['prim_energy_inc'],None,'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            energy_summation_par(molecule,msg['order'],molecule['corr_tuple'],molecule['corr_energy_inc'],None,'CORRE')
      #
      # remove_slave_env
      #
      elif (msg['task'] == 'remove_slave_env'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         # remove scr env
         #
         start_work = MPI.Wtime()
         #
         chdir(molecule['wrk'])
         #
         if (molecule['error'][-1]):
            #
            copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
         #
         rmtree(molecule['scr'],ignore_errors=True)
         #
         # collect mpi_time_work_slave
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
      #
      # collect_mpi_timings
      #
      elif (msg['task'] == 'collect_mpi_timings'):
         #
         # collect mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         collect_mpi_timings(molecule)
      #
      # finalize_mpi
      #
      elif (msg['task'] == 'finalize_mpi'):
         #
         slave = False
   #
   return molecule

