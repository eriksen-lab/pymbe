# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# energy-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import sys
import copy
from timeit import default_timer as timer
from mpi4py import MPI

import inc_corr_gen_rout
import inc_corr_orb_rout
import inc_corr_utils
import inc_corr_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def inc_corr_main(molecule):
   #
   # initialize domains
   #
   inc_corr_orb_rout.init_domains(molecule)
   #
   # initialize variable and lists
   #
   inc_corr_prepare(molecule)   
   #
   # run the specified calculation
   #
   if ((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')):
      #
      # run mono expansion
      #
      inc_corr_mono_exp(molecule)
      #
      # energy estimation for mono expansion
      #
      if (molecule['est']):
         #
         inc_corr_mono_exp_est(molecule,molecule['sec_tuple'][0],molecule['sec_domain'],molecule['sec_n_tuples'][0],molecule['sec_time'][0])
      #
      else:
         #
         for _ in range(0,len(molecule['e_tot'][0])):
            #
            molecule['sec_n_tuples'][0].append(0)
            #
            molecule['e_est'][0].append(0.0)
            #
            molecule['sec_time'][0].append(0.0)
   #
   elif ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      # run dual expansion
      #
      inc_corr_dual_exp(molecule)
   #
   return molecule

def inc_corr_mono_exp(molecule):
   #
   for k in range(1,molecule['max_order']+1):
      #
      # call mono expansion kernel
      #
      inc_corr_mono_exp_kernel(molecule,molecule['prim_tuple'][0],molecule['prim_domain'],molecule['prim_n_tuples'][0],molecule['prim_time'][0],k)
      #
      # print status end
      #
      inc_corr_utils.print_status_end(k,molecule['prim_time'][0],molecule['prim_n_tuples'][0],'MACRO')
      #
      # return if converged
      #
      if (molecule['conv'][-1]):
         #
         print('')
         print('')
         #
         return molecule
      #
      # orbital screening
      #
      if ((k >= 2) and (molecule['prim_thres'][0] != 0.0)):
         #
         inc_corr_orb_rout.orb_screen_rout(molecule,molecule['prim_tuple'][0],molecule['prim_orbital'][0],molecule['prim_domain'],\
                                           molecule['prim_thres'][0],molecule['l_limit'][0],molecule['u_limit'][0],'MACRO')
   #
   return molecule

def inc_corr_mono_exp_kernel(molecule,tup,dom,n_tup,time,k):
   #
   # define level
   #
   level = 'MACRO'
   #
   # generate all tuples at order k
   #
   tup.append([])
   #
   # print status header-1
   #
   inc_corr_utils.print_status_header_1(k)
   #
   # start time
   #
   start = timer()
   #
   inc_corr_orb_rout.orb_generator(molecule,dom,tup[k-1],molecule['l_limit'][0],k)
   #
   # collect time_gen
   #
   time_gen = timer() - start
   #
   # determine number of tuples at order k
   #
   n_tup.append(len(tup[k-1]))
   #
   # check for convergence
   #
   if (n_tup[-1] == 0):
      #
      molecule['conv'].append(True)
   #
   # calculate theoretical number of tuples at order k
   #
   inc_corr_orb_rout.n_theo_tuples(n_tup[0],k,molecule['theo_work'][0])
   #
   # print status header-2
   #
   inc_corr_utils.print_status_header_2(n_tup[k-1],k,molecule['conv'][-1],time_gen)
   #
   # return if converged
   #
   if (molecule['conv'][-1]):
      #
      for l in range(k+1,molecule['u_limit'][0]+1):
         #
         n_tup.append(0)
         #
         inc_corr_orb_rout.n_theo_tuples(n_tup[0],l,molecule['theo_work'][0])
      #
      return molecule
   #
   # start time
   #
   start = timer()
   #
   # run the calculations
   #
   if (molecule['mpi_parallel']):
      #
      energy_calc_mono_exp_par(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   else:
      #
      energy_calc_mono_exp_ser(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # calculate the energy at order k
   #
   inc_corr_order(molecule,k,n_tup,tup,molecule['e_tot'][0])
   #
   # collect time
   #
   time.append(timer()-start)
   #
   # print results
   #
   inc_corr_utils.print_result(tup[-1])
   #
   # check for convergence
   #
   if ((k == n_tup[0]) or (k == molecule['max_order'])):
      #
      tup.append([])
      #
      molecule['conv'].append(True)
   #
   return molecule

def inc_corr_mono_exp_est(molecule,tup,dom,n_tup,time):
   #
   # define level
   #
   level = 'ESTIM'
   #
   # set molecule['max_est_order']
   #
   diff_order = 0
   #
   for i in range(0,len(molecule['prim_n_tuples'][0])):
      #
      if ((molecule['prim_n_tuples'][0][i] < molecule['theo_work'][0][i]) and (molecule['prim_n_tuples'][0][i] > 0)):
         #
         diff_order = i
         #
         break
   #
   if (diff_order == 0):
      #
      molecule['max_est_order'] = 0
      #
      molecule['est_order'] = 0
      #
      for _ in range(0,len(molecule['e_tot'][0])):
         #
         n_tup.append(0)
         #
         molecule['e_est'][0].append(0.0)
         #
         time.append(0.0)
      #
      return molecule
   #
   elif ((diff_order + molecule['est_order']) > (len(molecule['prim_tuple'][0])-1)):
      #
      molecule['max_est_order'] = len(molecule['prim_tuple'][0])-1
      #
      molecule['est_order'] = (len(molecule['prim_tuple'][0])-1) - diff_order
   #
   else:
      #
      molecule['max_est_order'] = diff_order + molecule['est_order']
   #
   # generate all tuples required for energy estimation
   #
   # print status header-1
   #
   inc_corr_utils.print_status_header_est_1(molecule['max_est_order'])
   #
   # start time
   #
   start = timer()
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      # generate all tuples at order k
      #
      tup.append([])
      #
      inc_corr_orb_rout.orb_generator(molecule,dom,tup[k-1],molecule['l_limit'][0],k)
      #
      # only calculate required tuples at order k == molecule['max_est_order']
      #
      if (k == molecule['max_est_order']):
         #
         inc_corr_orb_rout.select_est_tuples(molecule['prim_tuple'][0],tup,k)
      #
      # determine number of tuples at order k
      #
      n_tup.append(len(tup[k-1]))
   #
   # collect time_gen
   #
   time_gen = timer() - start
   #
   # print status header-2
   #
   inc_corr_utils.print_status_header_est_2(molecule['max_est_order'],sum(n_tup),time_gen)
   #
   # init counter for STATUS-ESTIM
   #
   counter = 0
   #
   # perform energy estimation
   #
   string = ''
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      # start time
      #
      start = timer()
      #
      for i in range(0,n_tup[k-1]):
         #
         # increment counter
         #
         counter += 1
         #
         # write string
         #
         inc_corr_orb_rout.orb_string(molecule,molecule['l_limit'][0],molecule['u_limit'][0],tup[k-1][i][0],string)
         #
         # run correlated calc
         #
         inc_corr_gen_rout.run_calc_corr(molecule,string,level)
         #
         # write tuple energy
         #
         tup[k-1][i].append(molecule['e_tmp'])
         #
         # print status
         #
         inc_corr_utils.print_status(float(counter)/float(sum(n_tup)),level)
         #
         # error check
         #
         if (molecule['error'][0][-1]):
            #
            return molecule
      #
      # collect time
      #
      time.append(timer()-start)
   #
   # calculate the energies at all order k <= max_est_order
   #
   inc_corr_order_est(molecule,n_tup,tup,molecule['e_est'][0])
   #
   # print results
   #
   inc_corr_utils.print_result_est(molecule,tup)
   #
   # print status end
   #
   inc_corr_utils.print_status_end_est(molecule['max_est_order'],time)
   #
   # make the e_est and sec_time lists of the same length as the e_tot list
   #
   for _ in range(molecule['max_est_order'],len(molecule['e_tot'][0])):
      #
      molecule['e_est'][0].append(molecule['e_est'][0][-1])
      #
      time.append(0.0)
   #
   # make molecule['sec_n_tuples'] of the same length as molecule['prim_n_tuples']
   #
   for _ in range(molecule['max_est_order'],len(molecule['prim_n_tuples'][0])):
      #
      n_tup.append(0)
   #
   return molecule

def inc_corr_dual_exp(molecule):
   #
   for k in range(1,molecule['u_limit'][0]+1):
      #
      # append tuple list and generate all tuples at order k
      #
      molecule['tuple'][0].append([])
      #
      inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][0],molecule['tuple'][0][k-1],molecule['l_limit'][0],k)
      #
      # determine number of tuples at order k
      #
      molecule['n_tuples'][0].append(len(molecule['tuple'][0][k-1]))
      #
      # print status header (for outer expansion)
      #
      inc_corr_utils.print_status_header(molecule,molecule['n_tuples'][0],k)
      #
      # check for convergence (for outer expansion)
      #
      if (molecule['n_tuples'][0][k-1] == 0):
         #
         return molecule
      #
      # init time, energy diff, and relative work (for inner expansion)
      #
      molecule['time'][1][:] = []
      #
      molecule['e_diff_in'][:] = []
      #
      molecule['rel_work_in'][:] = []
      #
      # start time (for outer expansion)
      #
      start_out = timer()
      #
      # print result header (for outer expansion)
      #
      inc_corr_utils.print_result_header()
      #
      # run the calculations (for outer expansion)
      #
      for i in range(0,molecule['n_tuples'][0][k-1]):
         #
         molecule['e_tot'][1][:] = []
         #
         molecule['tuple'][1][:] = []
         #
         molecule['n_tuples'][1][:] = []
         #
         molecule['theo_work'][1][:] = []
         #
         # re-initialize the inner domain
         #
         inc_corr_orb_rout.reinit_domains(molecule,molecule['domain'][1])
         #
         # start time (for inner expansion)
         #
         start_in = timer()
         #
         for l in range(1,molecule['u_limit'][1]+1):
            #
            # append tuple list and generate all tuples at order l
            #
            molecule['tuple'][1].append([])
            #
            inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][1],molecule['tuple'][1][l-1],molecule['l_limit'][1],l)
            #
            # determine number of tuples at order l
            #
            molecule['n_tuples'][1].append(len(molecule['tuple'][1][l-1]))
            #
            # check for convergence (for inner expansion)
            #
            if (molecule['n_tuples'][1][l-1] == 0):
               #
               molecule['tuple'][0][k-1][i].append(molecule['e_tot'][1][-1])
               #
               inc_corr_utils.print_result(i,molecule['tuple'][0][k-1][i])
               #
               molecule['n_tuples'][1].pop()
               #
               break
            # 
            # run the calculations (for inner expansion)
            #
            string = ''
            #
            for j in range(0,molecule['n_tuples'][1][l-1]):
               #
               # write string
               #
               if (molecule['exp'] == 'comb-ov'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][0][k-1][i][0]+molecule['tuple'][1][l-1][j][0],string)
               #
               elif (molecule['exp'] == 'comb-vo'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][1][l-1][j][0]+molecule['tuple'][0][k-1][i][0],string)
               #
               # run correlated calc
               #
               inc_corr_gen_rout.run_calc_corr(molecule,string,False)
               #
               # write tuple energy
               #
               molecule['tuple'][1][l-1][j].append(molecule['e_tmp'])
               #
               # error check
               #
               if (molecule['error'][0][-1]):
                  #
                  return molecule
            #
            # calculate the energy at order l (for inner expansion)
            #
            inc_corr_order(l,molecule['tuple'][1],molecule['e_tot'][1])
            #
            # set up entanglement and exclusion lists (for inner expansion)
            #
            if (l >= 2):
               #
               molecule['orbital'][1].append([])
               #
               e_orb_rout(molecule,molecule['tuple'][1],molecule['orbital'][1],molecule['l_limit'][1],molecule['u_limit'][1])
               #
               molecule['excl_list'][1][:] = []
               #
               inc_corr_orb_rout.excl_rout(molecule,molecule['tuple'][1],molecule['orbital'][1],molecule['thres'][1],molecule['excl_list'][1])
               #
               # update domains (for inner expansion)
               #
               inc_corr_orb_rout.update_domains(molecule['domain'][1],molecule['l_limit'][1],molecule['excl_list'][1])
            #
            # calculate theoretical number of tuples at order l (for inner expansion)
            #
            inc_corr_orb_rout.n_theo_tuples(molecule['n_tuples'][1][0],l,molecule['theo_work'][1])
            #
            # check for maximum order (for inner expansion)
            #
            if (l == molecule['u_limit'][1]):
               #
               molecule['tuple'][0][k-1][i].append(molecule['e_tot'][1][-1])
               #
               inc_corr_utils.print_result(i,molecule['tuple'][0][k-1][i])
               #
               break
         #
         # collect time, energy diff, and relative work (for inner expansion)
         #
         molecule['time'][1].append(timer()-start_in)
         #
         molecule['e_diff_in'].append(molecule['e_tot'][1][-1]-molecule['e_tot'][1][-2])
         #
         molecule['rel_work_in'].append([])
         #
         for m in range(0,len(molecule['n_tuples'][1])):
            #
            molecule['rel_work_in'][-1].append((float(molecule['n_tuples'][1][m])/float(molecule['theo_work'][1][m]))*100.00)
            #
      #
      # print result end (for outer expansion)
      #
      inc_corr_utils.print_result_end()
      #
      # calculate the energy at order k (for outer expansion)
      #
      inc_corr_order(k,molecule['tuple'][0],molecule['e_tot'][0])
      #
      # set up entanglement and exclusion lists (for outer expansion)
      #
      if (k >= 2):
         #
         molecule['orbital'][0].append([])
         #
         e_orb_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         inc_corr_orb_rout.excl_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['thres'][0],molecule['excl_list'][0])
         #
         # update domains (for outer expansion)
         #
         inc_corr_orb_rout.update_domains(molecule['domain'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k (for outer expansion)
      #
      inc_corr_orb_rout.n_theo_tuples(molecule['n_tuples'][0][0],k,molecule['theo_work'][0])
      #
      # collect time (for outer expansion)
      #
      molecule['time'][0].append(timer()-start_out)
      #
      # print status end (for outer expansion)
      #
      inc_corr_utils.print_status_end(molecule,k,molecule['time'][0],molecule['n_tuples'][0])
      #
      # print results (for inner expansion)
      #
      inc_corr_utils.print_inner_result(molecule)
      #
      # print domain updates (for outer expansion)
      #
      if (k >= 2):
         #
         inc_corr_utils.print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
   #
   return molecule

def energy_calc_mono_exp_ser(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   for i in range(0,n_tup[order-1]):
      #
      # write string
      #
      inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
      #
      # run correlated calc
      #
      inc_corr_gen_rout.run_calc_corr(molecule,string['drop'],level)
      #
      # write tuple energy
      #
      tup[order-1][i].append(molecule['e_tmp'])
      #
      # print status
      #
      inc_corr_utils.print_status(float(i+1)/float(n_tup[order-1]),level)
      #
      # error check
      #
      if (molecule['error'][0][-1]):
         #
         return molecule, tup
   #
   return molecule, tup

def energy_calc_mono_exp_par(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size'] - 1
   #
   # number of available slaves
   #
   slaves_avail = num_slaves
   #
   # define mpi message tags
   #
   tags = inc_corr_utils.enum('ready','done','exit','start')
   #
   # init job index
   #
   i = 0
   #
   # init stat counter
   #
   counter = 0
   #
   # wake up slaves
   #
   msg = {'task': 'energy_calc_mono_exp'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[order-1]-1)):
         #
         inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
      #
      # run correlated calc
      #
      # receive data dict
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=molecule['mpi_stat'])
      #
      # probe for source
      #
      source = molecule['mpi_stat'].Get_source()
      #
      # probe for tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      if (tag == tags.ready):
         #
         if (i <= (n_tup[order-1]-1)):
            #
            # store job index
            #
            string['index'] = i
            #
            # send string dict
            #
            molecule['mpi_comm'].send(string, dest=source, tag=tags.start)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            molecule['mpi_comm'].send(None, dest=source, tag=tags.exit)
      #
      elif (tag == tags.done):
         #
         # write tuple energy
         #
         tup[order-1][data['index']].append(data['e_tmp'])
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         inc_corr_utils.print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' -- aborting...')
            #
            molecule['error'][0].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   return molecule, tup

def inc_corr_order(molecule,k,n_tup,tup,e_tot):
   #
   for j in range(0,n_tup[k-1]):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,n_tup[i-1]):
            #
            if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
               #
               tup[k-1][j][1] -= tup[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,n_tup[k-1]):
      #
      e_tmp += tup[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += e_tot[k-2]
   #
   e_tot.append(e_tmp)
   #
   return e_tot

def inc_corr_order_est(molecule,n_tup,tup,e_est):
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      for j in range(0,n_tup[k-1]):
         #
         for i in range(k-1,0,-1):
            #
            for l in range(0,n_tup[i-1]):
               #
               if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
                  #
                  tup[k-1][j][1] -= tup[i-1][l][1]
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      e_tmp = 0.0
      #
      for j in range(0,n_tup[k-1]):
         #
         found = False
         #
         for l in range(0,molecule['prim_n_tuples'][0][k-1]):
            #
            if (set(tup[k-1][j][0]) == set(molecule['prim_tuple'][0][k-1][l][0])):
               #
               found = True
               #
               break
         #
         if (not found):
            #
            e_tmp += tup[k-1][j][1]
      #
      if (k > 1):
         #
         e_tmp += e_est[k-2]
      #
      e_est.append(e_tmp)
   #
   return e_est

def inc_corr_prepare(molecule):
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['l_limit'] = [0]
      molecule['u_limit'] = [molecule['nocc']]
      #
      molecule['prim_domain'] = copy.deepcopy(molecule['occ_domain'])
      molecule['sec_domain'] = copy.deepcopy(molecule['occ_domain'])
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['prim_domain'] = copy.deepcopy(molecule['virt_domain'])
      molecule['sec_domain'] = copy.deepcopy(molecule['virt_domain'])
      #
   #
   elif (molecule['exp'] == 'comb-ov'):
      #
      molecule['l_limit'] = [0,molecule['nocc']]
      molecule['u_limit'] = [molecule['nocc'],molecule['nvirt']]
      #
      molecule['domain'] = [molecule['occ_domain'],molecule['virt_domain']]
      #
      molecule['e_diff_in'] = []
      #
      molecule['rel_work_in'] = []
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      molecule['l_limit'] = [molecule['nocc'],0]
      molecule['u_limit'] = [molecule['nvirt'],molecule['nocc']]
      #
      molecule['domain'] = [molecule['virt_domain'],molecule['occ_domain']]
      #
      molecule['e_diff_in'] = []
      #
      molecule['rel_work_in'] = []
   #
   if ((molecule['max_order'] == 0) or (molecule['max_order'] > molecule['u_limit'][0])):
      #
      molecule['max_order'] = molecule['u_limit'][0]
   #
   molecule['conv'] = [False]
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['prim_tuple'] = [[],[]]
   molecule['sec_tuple'] = [[],[]]
   #
   molecule['prim_n_tuples'] = [[],[]]
   molecule['sec_n_tuples'] = [[],[]]
   #
   molecule['prim_orbital'] = [[],[]]
   molecule['sec_orbital'] = [[],[]]
   #
   molecule['e_tot'] = [[],[]]
   #
   molecule['e_est'] = [[],[]]
   #
   molecule['excl_list'] = [[],[]]
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['prim_time'] = [[],[]]
   molecule['sec_time'] = [[],[]]
   #
   return molecule

