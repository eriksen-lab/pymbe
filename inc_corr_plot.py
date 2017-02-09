#!/usr/bin/env python

#
# python plotting program for inc.-corr. calculations 
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import seaborn as sns

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def abs_energy_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total '+molecule['model'].upper()+' energy')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      styles = ['--','-.','-','--','-.','-','--','-.','-','--','-.']
      #
      e_corr = []
      #
      for i in range(0,molecule['corr_order']):
         #
         e_corr.append([])
         #
         if (i == 0):
            #
            e_corr[i] = copy.deepcopy(molecule['e_tot'][0])
         #
         else:
            #
            e_corr[i] = copy.deepcopy(e_corr[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(e_corr[i])):
            #
            e_corr[i][j] += (molecule['e_corr'][0][(molecule['min_corr_order']+i)-1]-molecule['e_corr'][0][(molecule['min_corr_order']+i)-2])
   #
   ax.plot(list(range(1,len(molecule['e_tot'][0])+1)),molecule['e_tot'][0],marker='x',linewidth=2,linestyle='-',\
           label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_corr[i],\
                 marker='x',linestyle=styles[i],linewidth=2,label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax.set_xlim([0.5,u_limit+0.5])
   #
   ax.xaxis.grid(False)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Energy (in Hartree)')
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   sns.despine()
   #
   with sns.axes_style("whitegrid"):
      #
      insert = plt.axes([.35, .50, .50, .30],frameon=True)
      #
      insert.plot(list(range(2,len(molecule['e_tot'][0])+1)),molecule['e_tot'][0][1:],marker='x',linewidth=2,linestyle='-')
      #
      if (molecule['corr']):
         #
         for i in range(0,molecule['corr_order']):
            #
            insert.plot(list(range(2,len(molecule['e_tot'][0])+1)),e_corr[i][1:],\
                     marker='x',linewidth=2,linestyle=styles[i])
      #
      plt.setp(insert,xticks=list(range(3,len(molecule['e_tot'][0])+1)))
      #
      insert.set_xlim([2.5,len(molecule['e_tot'][0])+0.5])
      #
      insert.locator_params(axis='y',nbins=6)
      #
      insert.set_ylim([molecule['e_tot'][0][-1]-0.01,\
                       molecule['e_tot'][0][-1]+0.01])
      #
      insert.xaxis.grid(False)
   #
   if (molecule['corr']):
      #
      ax.legend(loc=1,ncol=molecule['corr_order']+1)
   #
   else:
      #
      ax.legend(loc=1)
   #
   plt.savefig(molecule['wrk']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def n_tuples_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total number of '+molecule['model'].upper()+' tuples')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      corr_prim = []
      theo_corr = []
      #
      for i in range(0,u_limit):
         #
         corr_prim.append(molecule['corr_n_tuples'][0][i])
         theo_corr.append(molecule['theo_work'][0][i]-(molecule['prim_n_tuples'][0][i]+molecule['corr_n_tuples'][0][i]))
      #
      sns.barplot(list(range(1,u_limit+1)),theo_corr,bottom=[(i + j) for i,j in zip(corr_prim,molecule['prim_n_tuples'][0])],\
                  palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),corr_prim,bottom=molecule['prim_n_tuples'][0],palette='Reds_r',\
                  label='Energy corr.',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'][0],palette='Blues_r',\
                  label='BG('+molecule['model'].upper()+') expansion',log=True)
   #
   else:
      #
      sns.barplot(list(range(1,u_limit+1)),[(i - j) for i,j in zip(molecule['theo_work'][0],molecule['prim_n_tuples'][0])],\
                  bottom=molecule['prim_n_tuples'][0],palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'][0],palette='Blues_r',\
                  label='BG('+molecule['model'].upper()+') expansion',log=True)
   #
   ax.xaxis.grid(False)
   #
   ax.set_ylim(bottom=0.7)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Number of correlated tuples')
   #
   plt.legend(loc=1)
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/n_tuples_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def orb_entangl_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   tot_plots = len(molecule['prim_orbital'][0])
   tot_cols = 1
   #
   tot_rows = tot_plots // tot_cols
   tot_rows += tot_plots % tot_cols
   #
   position = range(1,tot_plots+1)
   #
   print('tot_plots = {0:} , tot_cols = {1:} , tot_rows = {2:}'.format(tot_plots,tot_cols,tot_rows))
   #
   arr_list = []
   #
   h_map = []
   #
   for i in range(0,tot_plots):
      #
      arr_list[:] = []
      #
      for j in range(0,len(molecule['prim_orbital'][0][i])):
         #
         arr_list.append([])
         #
         for k in range(0,len(molecule['prim_orbital'][0][i][j])-1):
            #
            arr_list[j].append(molecule['prim_orbital'][0][i][j][k+1][1][1]*100.0)
      #
      orbital_arr = np.asarray(arr_list)
      #
      h_map.append(fig.add_subplot(tot_rows,tot_cols,position[i]))
      #
      h_map[i] = sns.heatmap(orbital_arr,linewidths=.5,xticklabels=range(molecule['l_limit'][0]+1,(molecule['l_limit'][0]+molecule['u_limit'][0])+1),\
                       yticklabels=range(1,len(molecule['e_tot'][0])+1),cmap='coolwarm',cbar=False,\
                       annot=True,fmt='.1f',vmin=-np.amax(orbital_arr),vmax=np.amax(orbital_arr))
      #
      h_map[i].set_title('orbital entanglement matrix (k = '+str(i+2))
      #
#      ax.set_xlabel('Orbital')
#      ax.set_ylabel('Order')
   #
   plt.yticks(rotation=0)
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/orb_entangl_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def dev_ref_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   sns.despine()
   #
   fig, ( ax1, ax2 ) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   kcal_mol = 0.001594
   #
   if (molecule['corr']):
      #
      styles = ['--','-.','-','--','-.','-','--','-.','-','--','-.']
      #
      e_corr = []
      #
      for i in range(0,molecule['corr_order']):
         #
         e_corr.append([])
         #
         if (i == 0):
            #
            e_corr[i] = copy.deepcopy(molecule['e_tot'][0])
         #
         else:
            #
            e_corr[i] = copy.deepcopy(e_corr[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(e_corr[i])):
            #
            e_corr[i][j] += (molecule['e_corr'][0][(molecule['min_corr_order']+i)-1]-molecule['e_corr'][0][(molecule['min_corr_order']+i)-2])
   #
   e_diff_tot_abs = []
   e_diff_corr_abs = []
   e_diff_tot_rel = []
   e_diff_corr_rel = []
   #
   for j in range(0,len(molecule['e_tot'][0])):
      #
      e_diff_tot_abs.append((molecule['e_tot'][0][j]-molecule['e_ref'])/kcal_mol)
      e_diff_tot_rel.append((molecule['e_tot'][0][j]/molecule['e_ref'])*100.)
      #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         e_diff_corr_abs.append([])
         e_diff_corr_rel.append([])
         #
         for j in range(0,len(molecule['e_tot'][0])):
            #
            e_diff_corr_abs[i].append((e_corr[i][j]-molecule['e_ref'])/kcal_mol)
            e_diff_corr_rel[i].append((e_corr[i][j]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model'].upper()+')')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_tot_abs,marker='x',linewidth=2,linestyle='-',\
            label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax1.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_corr_abs[i],marker='x',linewidth=2,linestyle=styles[i],\
                  label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax1.set_ylim([-3.4,3.4])
   #
   ax1.xaxis.grid(False)
   #
   ax1.set_ylabel('Difference (in kcal/mol)')
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax1.legend(loc=1)
   #
   ax2.set_title('Relative recovery of E('+molecule['model'].upper()+')')
   #
   ax2.axhline(100.0,color='black',linewidth=2)
   #
   ax2.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-',\
            label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax2.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_corr_rel[i],marker='x',linewidth=2,linestyle=styles[i],\
                  label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax2.set_xlim([0.5,u_limit+0.5])
   #
   ax2.xaxis.grid(False)
   #
   ax2.set_ylim([93.,107.])
   #
   ax2.set_ylabel('Recovery (in %)')
   ax2.set_xlabel('Expansion order')
   #
   ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax2.legend(loc=1)
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/dev_ref_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def ic_plot(molecule):
   #
   #  ---  plot total energies  ---
   #
   abs_energy_plot(molecule)
   #
   #  ---  plot number of calculations from each orbital  ---
   #
   n_tuples_plot(molecule)
   #
   #  ---  plot orbital entanglement matrices  ---
   #
#   orb_entangl_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']):
      #
      dev_ref_plot(molecule)
   #
   return molecule


