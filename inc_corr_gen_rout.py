#!/usr/bin/env python

#
# generel, yet specific routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import os
import re
from timeit import default_timer as timer

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def run_calc_hf(molecule):
   #
   write_zmat_hf(molecule)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   get_dim(molecule)
   #
   if (not molecule['error'][0][-1]):
      #
      command='xclean'
      os.system(command)
   #
   return molecule

def run_calc_corr(molecule,drop_string,level):
   #
   write_zmat_corr(molecule,drop_string,level)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   write_energy(molecule,level)
   #
   if (not molecule['error'][0][-1]):
      command='xclean'
      os.system(command)
   #
   return molecule

def write_zmat_hf(molecule):
   #
   out=open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   out.write('*CFOUR(CALC=HF\n')
   out.write('SCF_CONV=10\n')
   out.write('LINEQ_CONV=9\n')
   #
   if (not molecule['zmat']):
      #
      out.write('COORD=CARTESIAN\n')
   #
   if (molecule['units'] == 'bohr'):
      #
      out.write('UNITS=BOHR\n')
   #
   out.write('MULTIPLICITY='+str(molecule['mult'])+'\n')
   #
   if (molecule['mult'] == 1):
      #
      out.write('REF=RHF\n')
   #
   else:
      #
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+molecule['basis'].upper()+'\n')
   #
   out.write('MEMORY='+str(molecule['mem'])+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def write_zmat_corr(molecule,drop_string,level):
   #
   out=open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   if (level == 'ESTIM'):
      #
      model = molecule['est_model'].upper()
   #
   else:
      #
      model = molecule['model'].upper()
   #
   if (model.lower() == 'fci'):
      #
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
      out.write('CAS_MITMAX=200\n')
   #
   else:
      #
      out.write('*CFOUR(CALC='+model+'\n')
      out.write('CC_PROG=VCC\n')
      out.write('CC_EXPORDER=10\n')
      out.write('CC_MAXCYC=200\n')
   #
   if (drop_string != '\n'):
      #
      out.write(drop_string)
   #
   out.write('SCF_CONV=10\n')
   out.write('LINEQ_CONV=9\n')
   out.write('CC_CONV=9\n')
   #
   if (not molecule['zmat']):
      #
      out.write('COORD=CARTESIAN\n')
   #
   if (molecule['units'] == 'bohr'):
      #
      out.write('UNITS=BOHR\n')
   #
   if (molecule['local']):
      #
      out.write('SYMMETRY=OFF\n')
      out.write('ORBITALS=LOCAL\n')
   #
   if ((molecule['frozen'] == 'conv') and (level == 'REF')):
      #
      out.write('FROZEN_CORE=ON\n')
   #
   out.write('MULTIPLICITY='+str(molecule['mult'])+'\n')
   #
   if (molecule['mult'] == 1):
      #
      out.write('REF=RHF\n')
   #
   else:
      #
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+molecule['basis'].upper()+'\n')
   #
   out.write('MEMORY='+str(molecule['mem'])+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()
   #
   return molecule

def get_dim(molecule):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   regex = 'basis functions'
   #
   while 1:
      #
      line=inp.readline()
      #
      if regex in line:
         #
         [bf] = line.split()[2:3]
         break
      #
      elif re.match(regex_err,line) is not None:
         #
         print('problem with HF calculation, aborting ...')
         molecule['error'][0].append(True)
         inp.close()
         return molecule
   #
   inp.seek(0)
   #
   regex_2 = '\s+Alpha population by irrep:'
   #
   while 1:
      #
      line=inp.readline()
      #
      if re.match(regex_2,line) is not None:
         #
         pop = line.split()
         break
   #
   tmp = 0
   #
   for i in range(4,len(pop)):
      tmp += int(pop[i])
   #
   molecule['nocc'] = tmp
   molecule['nvirt'] = int(bf) - molecule['nocc']
   #
   inp.close()
   #
   return molecule

def write_energy(molecule,level):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   if ((level == 'MACRO') or (level == 'REF')):
      #
      model = molecule['model']
      regex = molecule['regex']
   #
   elif (level == 'ESTIM'):
      #
      model = molecule['est_model']
      regex = molecule['est_regex']
   #
   while 1:
      #
      line=inp.readline()
      #
      if re.match(regex,line) is not None:
         #
         if (model == 'fci'):
            #
            [tmp] = line.split()[3:4]
         #
         elif (model == 'mp2'):
            #
            [tmp] = line.split()[2:3]
         #
         else: # CC
            #
            [tmp] = line.split()[4:5]
         #
         if (level == 'REF'):
            #
            molecule['e_ref'] = float(tmp)
         #
         else:
            #
            molecule['e_tmp'] = float(tmp)
         #
         break
      #
      elif re.match(regex_err,line) is not None:
         #
         print('problem with '+model+' calculation, aborting ...')
         molecule['error'][0].append(True)
         inp.close()
         #
         return molecule
   #
   inp.close()
   #
   return molecule

def ref_calc(molecule):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-REF:  full reference calculation started')
   print(' --------------------------------------------------------------------------------------------')
   #
   start = timer()
   #
   run_calc_corr(molecule,'','REF')
   #
   molecule['prim_time'][0].append(timer()-start)
   #
   print(' STATUS-REF:  full reference calculation done in {0:10.2e} seconds'.format(molecule['prim_time'][0][-1]))
   print(' --------------------------------------------------------------------------------------------')
   print('')
   #
   return molecule

