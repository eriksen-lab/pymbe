#!/usr/bin/env python
# -*- coding: utf-8 -*

""" results.py: summary and plotting module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import contextlib
import numpy as np
from pyscf import symm
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
try:
	import seaborn as sns
	SNS_FOUND = True
except (ImportError, OSError):
	pass
	SNS_FOUND = False

import output
import tools


# results parameters
DIVIDER = '{:^143}'.format('-'*137)
FILL = '{:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
		""" printing and plotting of results """
		# setup
		info = _setup(mpi, mol, calc, exp)
		# results
		_table(info, mpi, mol, calc, exp)
		# plot
		_plot(info, calc, exp)


def _setup(mpi, mol, calc, exp):
		""" parameters """
		# init info dict
		info = {}
		# parameters
		info['model'] = _model(calc)
		info['basis'] = _basis(mol)
		info['state'] = _state(mol, calc)
		info['ref'] = _ref(mol, calc)
		info['base'] = _base(calc)
		info['prot'] = _prot(calc)
		info['system'] = _system(mol, calc)
		info['frozen'] = _frozen(mol)
		if mol.atom:
			info['hubbard'] = None
		else:
			info['hubbard'] = _hubbard(mol)
		info['solver'] = _solver(calc)
		info['active'] = _active(calc)
		info['orbs'] = _orbs(calc)
		info['mpi'] = _mpi(mpi, calc)
		info['thres'] = _thres(calc)
		info['symm'] = _symm(mol, calc)
		if calc.target == 'energy':
			info['energy'] = _energy(calc, exp)
		else:
			info['energy'] = None
		if calc.target == 'excitation':
			info['excitation'] = _excitation(calc, exp)
		else:
			info['excitation'] = None
		if calc.target == 'dipole':
			info['dipole'], info['nuc_dipole'] = _dipole(mol, calc, exp)
		else:
			info['dipole'] = info['nuc_dipole'] = None
		if calc.target == 'trans':
			info['trans'] = _trans(mol, calc, exp)
		else:
			info['trans'] = None
		info['final_order'] = exp.order + 1
		return info


def _table(info, mpi, mol, calc, exp):
		""" print results """
		# print header
		print(output.main_header())
		# write results to results.out
		print(_summary_prt(info, mol, calc, exp))
		print(_timings_prt(info, calc, exp))
		if calc.target == 'energy' :
			print(_energy_prt(info, calc, exp))
		if calc.target == 'excitation':
			print(_excitation_prt(info, calc, exp))
		if calc.target == 'dipole' :
			print(_dipole_prt(info, calc, exp))
		if calc.target == 'trans':
			print(_trans_prt(info, calc, exp))
	

def _plot(info, calc, exp):
		""" plot results """
		# plot MBE quantitites
		if calc.target == 'energy':
			_energies_plot(info, calc, exp)
		elif calc.target == 'excitation':
			_excitation_plot(info, calc, exp)
		elif calc.target == 'dipole':
			_dipole_plot(info, calc, exp)
		elif calc.target == 'trans':
			_trans_plot(info, calc, exp)
			_osc_strength_plot(info, calc, exp)


def _model(calc):
		""" model print """
		return '{:}'.format(calc.model['method'].upper())


def _basis(mol):
		""" basis print """
		if isinstance(mol.basis, str):
			return mol.basis
		elif isinstance(mol.basis, dict):
			for i, val in enumerate(mol.basis.items()):
				if i == 0:
					basis = val[1]
				else:
					basis += '/'+val[1]
			return basis


def _state(mol, calc):
		""" state print """
		string = '{:}'.format(calc.state['root'])
		if mol.spin == 0:
			string += ' (singlet)'
		elif mol.spin == 1:
			string += ' (doublet)'
		elif mol.spin == 2:
			string += ' (triplet)'
		elif mol.spin == 3:
			string += ' (quartet)'
		elif mol.spin == 4:
			string += ' (quintet)'
		else:
			string += ' ({:})'.format(mol.spin+1)
		return string


def _ref(mol, calc):
		""" ref print """
		if calc.ref['method'] == 'casci':
			return 'CASCI'
		elif calc.ref['method'] == 'casscf':
			if len(calc.ref['wfnsym']) == 1:
				return 'CASSCF'
			else:
				for i in range(len(set(calc.ref['wfnsym']))):
					sym = symm.addons.irrep_id2name(mol.symmetry, list(set(calc.ref['wfnsym']))[i])
					num = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[i])
					if i == 0:
						syms = str(num)+'*'+sym
					else:
						syms += '/'+sym
				return 'CASSCF('+syms+')'


def _base(calc):
		""" base print """
		if calc.base['method'] is None:
			return 'none'
		else:
			return calc.base['method'].upper()


def _prot(calc):
		""" protocol print """
		return calc.prot['scheme']


def _system(mol, calc):
		""" system size print """
		return '{:} e in {:} o'.format(mol.nelectron - 2 * mol.ncore, calc.ref_space.size + calc.exp_space.size)


def _hubbard(mol):
		""" hubbard print """
		hubbard = ['{:} x {:}'.format(mol.matrix[0], mol.matrix[1])]
		hubbard.append('{:} & {:}'.format(mol.u, mol.n))
		return hubbard


def _solver(calc):
		""" FCI solver print """
		if calc.model['method'] != 'fci':
			return 'none'
		else:
			if calc.model['solver'] == 'pyscf_spin0':
				return 'PySCF (spin0)'
			elif calc.model['solver'] == 'pyscf_spin1':
				return 'PySCF (spin1)'


def _frozen(mol):
		""" frozen core print """
		if mol.frozen:
			return 'true'
		else:
			return 'false'


def _active(calc):
		""" active space print """
		return '{:} e in {:} o'.format(calc.nelec[0] + calc.nelec[1], calc.ref_space.size)


def _orbs(calc):
		""" orbital print """
		if calc.orbs['type'] in ['canonical', 'dynamic']:
			return calc.orbs['type']
		elif calc.orbs['type'] == 'ccsd':
			return 'CCSD NOs'
		elif calc.orbs['type'] == 'ccsd(t)':
			return 'CCSD(T) NOs'
		elif calc.orbs['type'] == 'local':
			return 'pipek-mezey'


def _mpi(mpi, calc):
		""" mpi print """
		return '{:} & {:}'.format(calc.mpi['masters'], mpi.size - calc.mpi['masters'])


def _thres(calc):
		""" threshold print """
		return '{:.0e} ({:<.1f})'.format(calc.thres['init'], calc.thres['relax'])


def _symm(mol, calc):
		""" symmetry print """
		if calc.model['method'] == 'fci':
			if mol.atom:
				string = symm.addons.irrep_id2name(mol.symmetry, calc.state['wfnsym'])+'('+mol.symmetry+')'
				if calc.extra['sigma']:
					string += '*Sigma'
				return string
			else:
				return 'C1(A)'
		else:
			return 'unknown'


def _energy(calc, exp):
		""" final energies """
		return exp.prop['energy']['tot'] \
				+ calc.prop['hf']['energy'] \
				+ calc.prop['base']['energy'] \
				+ calc.prop['ref']['energy']


def _excitation(calc, exp):
		""" final energies """
		return exp.prop['excitation']['tot'] \
				+ calc.prop['ref']['excitation']


def _dipole(mol, calc, exp):
		""" final molecular dipole moments """
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipole = np.einsum('i,ix->x', charges, coords)
		dipole = exp.prop['dipole']['tot'] \
						+ calc.prop['hf']['dipole'] \
						+ calc.prop['ref']['dipole']
		return dipole, nuc_dipole


def _trans(mol, calc, exp):
		""" final molecular transition dipole moments """
		return exp.prop['trans']['tot'] \
				+ calc.prop['ref']['trans']


def _time(exp, comp, idx):
		""" convert time to (HHH : MM : SS) format """
		# init time
		if comp in ['mbe', 'screen']:
			time = exp.time[comp][idx]
		elif comp == 'sum':
			time = exp.time['mbe'][idx] + exp.time['screen'][idx]
		elif comp in ['tot_mbe', 'tot_screen']:
			time = np.sum(exp.time[comp[4:]])
		elif comp == 'tot_sum':
			time = np.sum(exp.time['mbe']) + np.sum(exp.time['screen'])
		hours = int(time // 3600)
		minutes = int((time - (time // 3600) * 3600.)//60)
		seconds = time - hours * 3600. - minutes * 60.
		# init time string
		time_str = ''
		if hours > 0:
			time_str += '{:}h '.format(hours)
		if minutes > 0:
			time_str += '{:}m '.format(minutes)
		time_str += '{:.2f}s'.format(seconds)
		return time_str


def _summary_prt(info, mol, calc, exp):
		""" summary table """
		string = DIVIDER+'\n'
		string += '{:14}{:21}{:12}{:1}{:12}{:21}{:11}{:1}{:13}{:}\n'
		form = ('','molecular information','','|','', \
					'expansion information','','|','','calculation information',)
		string += DIVIDER+'\n'
		if mol.atom:
			string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
						'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
			form += ('','basis set','','=','',info['basis'], \
						'','|','','exp. model','','=','',info['model'], \
						'','|','','mpi masters & slaves','','=','',info['mpi'],)
			string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
					'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
			form += ('','frozen core','','=','',info['frozen'], \
						'','|','','ref. function','','=','',info['ref'], \
						'','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy'],)
		else:
			string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
					'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
			form += ('','hubbard matrix','','=','',info['hubbard'][0], \
						'','|','','exp. model','','=','',info['model'], \
						'','|','','mpi masters & slaves','','=','',info['mpi'],)
			string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
					'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
			form += ('','hubbard U/t & n','','=','',info['hubbard'][1], \
						'','|','','ref. function','','=','',info['ref'], \
						'','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy'],)
		string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
				'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
		form += ('','system size','','=','',info['system'], \
					'','|','','exp. reference','','=','',info['active'], \
					'','|','','base model energy','','=','', \
					calc.prop['hf']['energy']+calc.prop['base']['energy'],)
		string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
				'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
		form += ('','state (mult.)','','=','',info['state'], \
					'','|','','base model','','=','',info['base'], \
					'','|','','MBE total energy','','=','', \
					calc.prop['hf']['energy'] if info['energy'] is None else info['energy'][-1],)
		string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
				'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
		form += ('','orbitals','','=','',info['orbs'], \
					'','|','','screen. prot.','','=','',info['prot'], \
					'','|','','total time','','=','',_time(exp, 'tot_sum', -1),)
		string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
				'{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
		form += ('','FCI solver','','=','',info['solver'], \
					'','|','','screen. thres.','','=','',info['thres'], \
					'','|','','wave funct. symmetry','','=','',info['symm'],)
		string += DIVIDER+'\n'+FILL+'\n'+DIVIDER+'\n'
		return string.format(*form)


def _timings_prt(info, calc, exp):
		""" timings """
		string = DIVIDER[:98]+'\n'
		string += '{:^98}\n'
		form = ('MBE timings',)
		string += DIVIDER[:98]+'\n'
		string += '{:6}{:9}{:2}{:1}{:8}{:3}{:8}{:1}{:5}{:9}{:5}' \
				'{:1}{:8}{:3}{:8}{:1}{:5}{:}\n'
		form += ('','MBE order','','|','','MBE','','|','','screening', \
					'','|','','sum','','|','','calculations',)
		string += DIVIDER[:98]+'\n'
		calcs = 0
		for i, j in enumerate(range(1, info['final_order'])):
			calc_i = exp.count[i]
			calcs += calc_i
			string += '{:7}{:>4d}{:6}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}{:1}' \
					'{:2}{:>15s}{:2}{:1}{:5}{:>9d}\n'
			form += ('',j, \
						'','|','',_time(exp, 'mbe', i), \
						'','|','',_time(exp, 'screen', i), \
						'','|','',_time(exp, 'sum', i), \
						'','|','',calc_i,)
		string += DIVIDER[:98]+'\n'
		string += '{:8}{:5s}{:4}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}{:1}' \
				'{:2}{:>15s}{:2}{:1}{:5}{:>9d}\n'
		form += ('','total', \
					'','|','',_time(exp, 'tot_mbe', -1), \
					'','|','',_time(exp, 'tot_screen', -1), \
					'','|','',_time(exp, 'tot_sum', -1), \
					'','|','',calcs,)
		string += DIVIDER[:98]+'\n'
		return string.format(*form)


def _energy_prt(info, calc, exp):
		""" energies """
		string = DIVIDER[:66]+'\n'
		string_in = 'MBE energy (root = '+str(calc.state['root'])+')'
		string += '{:^66}\n'
		form = (string_in,)
		string += DIVIDER[:66]+'\n'
		string += '{:6}{:9}{:2}{:1}{:5}{:12}{:5}{:1}{:4}{:}\n'
		form += ('','MBE order','','|','','total energy','','|','','correlation energy',)
		string += DIVIDER[:66]+'\n'
		string += '{:9}{:>3s}{:5}{:1}{:5}{:>11.6f}{:6}{:1}{:7}{:11.4e}\n'
		form += ('','ref','','|','',calc.prop['hf']['energy'] + calc.prop['ref']['energy'],'','|','',calc.prop['ref']['energy'],)
		string += DIVIDER[:66]+'\n'
		for i, j in enumerate(range(1, info['final_order'])):
			string += '{:7}{:>4d}{:6}{:1}{:5}{:>11.6f}{:6}{:1}{:7}{:11.4e}\n'
			form += ('',j, \
						'','|','',info['energy'][i], \
						'','|','',info['energy'][i] - calc.prop['hf']['energy'],)
		string += DIVIDER[:66]+'\n'
		return string.format(*form)


def _energies_plot(info, calc, exp):
		""" plot MBE energy """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy
		energy = info['energy'].copy()
		# plot results
		ax1.plot(np.arange(1, info['final_order']), \
				energy, marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total energy increments
		mbe = exp.prop['energy']['tot'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.arange(1, info['final_order']), \
				np.abs(mbe), marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(tools.OUT+'/energy_state_{:}.pdf'.format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _excitation_prt(info, calc, exp):
		""" excitation energies """
		string = DIVIDER[:43]+'\n'
		string_in = 'MBE excitation energy (root = '+str(calc.state['root'])+')'
		string += '{:^46}\n'
		form = (string_in,)
		string += DIVIDER[:43]+'\n'
		string += '{:6}{:9}{:2}{:1}{:5}{:}\n'
		form += ('','MBE order','','|','','excitation energy',)
		string += DIVIDER[:43]+'\n'
		string += '{:9}{:>3s}{:5}{:1}{:8}{:9.4e}\n'
		form += ('','ref','','|','',calc.prop['ref']['excitation'],)
		string += DIVIDER[:43]+'\n'
		for i, j in enumerate(range(1, info['final_order'])):
			string += '{:7}{:>4d}{:6}{:1}{:8}{:9.4e}\n'
			form += ('',j,'','|','',info['excitation'][i],)
		string += DIVIDER[:43]+'\n'
		return string.format(*form)


def _excitation_plot(info, calc, exp):
		""" plot MBE excitation energy """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy
		excitation = info['excitation'].copy()
		# plot results
		ax1.plot(np.arange(1, info['final_order']), \
				excitation, marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Excitation energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total energy increments
		mbe = exp.prop['excitation']['tot'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.arange(1, info['final_order']), \
				np.abs(mbe), marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(tools.OUT+'/excitation_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _dipole_prt(info, calc, exp):
		""" dipole moments """
		string = DIVIDER[:82]+'\n'
		string_in = 'MBE dipole moment (root = '+str(calc.state['root'])+')'
		string += '{:^82}\n'
		form = (string_in,)
		string += DIVIDER[:82]+'\n'
		string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
		form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)
		string += DIVIDER[:82]+'\n'
		string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
		form += ('','ref', \
					'','|','',info['nuc_dipole'][0] - calc.prop['hf']['dipole'][0] + calc.prop['ref']['dipole'][0], \
					'',info['nuc_dipole'][1] - calc.prop['hf']['dipole'][1] + calc.prop['ref']['dipole'][1], \
					'',info['nuc_dipole'][2] - calc.prop['hf']['dipole'][2] + calc.prop['ref']['dipole'][2], \
					'','|','',np.linalg.norm(info['nuc_dipole'] - calc.prop['hf']['dipole']),)
		string += DIVIDER[:82]+'\n'
		for i, j in enumerate(range(1, info['final_order'])):
			string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
			form += ('',j, \
						'','|','',info['nuc_dipole'][0] - info['dipole'][i, 0], \
						'',info['nuc_dipole'][1] - info['dipole'][i, 1], \
						'',info['nuc_dipole'][2] - info['dipole'][i, 2], \
						'','|','',np.linalg.norm(info['nuc_dipole'] - info['dipole'][i, :]),)
		string += DIVIDER[:82]+'\n'
		return string.format(*form)


def _dipole_plot(info, calc, exp):
		""" plot MBE dipole moment """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE dipole moment
		dipole = np.empty(info['dipole'].shape[0], dtype=np.float64)
		for i in range(dipole.size):
			dipole[i] = np.linalg.norm(info['nuc_dipole'] - info['dipole'][i, :])
		# plot results
		ax1.plot(np.arange(1, info['final_order']), \
				dipole, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Dipole moment (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total dipole increments
		mbe = np.empty_like(dipole)
		for i in range(mbe.size):
			mbe[i] = np.linalg.norm(exp.prop['dipole']['tot'][i, :])
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.arange(1, info['final_order']), \
				np.abs(mbe), marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(tools.OUT+'/dipole_state_{:}.pdf'.format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _trans_prt(info, calc, exp):
		""" transition dipole moments """
		string = DIVIDER[:109]+'\n'
		string_in = 'MBE transition dipole moment (excitation 0 > '+str(calc.state['root'])+')'
		string += '{:^109}\n'
		form = (string_in,)
		string += DIVIDER[:109]+'\n'
		string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:13}{:3}{:1}{:4}{:}\n'
		form += ('','MBE order','','|','','dipole components (x,y,z)', \
					'','|','','dipole moment','','|','','oscillator strength',)
		string += DIVIDER[:109]+'\n'
		string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}{:6}{:1}{:8}{:9.6f}\n'
		form += ('','ref', \
					'','|','',calc.prop['ref']['trans'][0], \
					'',calc.prop['ref']['trans'][1], \
					'',calc.prop['ref']['trans'][2], \
					'','|','',np.linalg.norm(calc.prop['ref']['trans'][:]), \
					'','|','',(2./3.) * calc.prop['ref']['excitation'] * np.linalg.norm(calc.prop['ref']['trans'][:])**2,)
		string += DIVIDER[:109]+'\n'
		for i, j in enumerate(range(1, info['final_order'])):
			string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}{:6}{:1}{:8}{:9.6f}\n'
			form += ('',j, \
						'','|','',info['trans'][i, 0], \
						'',info['trans'][i, 1], \
						'',info['trans'][i, 2], \
						'','|','',np.linalg.norm(info['trans'][i, :]), \
						'','|','',(2./3.) * info['excitation'][i] * np.linalg.norm(info['trans'][i, :])**2,)
		string += DIVIDER[:109]+'\n'
		return string.format(*form)


def _trans_plot(info, calc, exp):
		""" plot MBE transition dipole moments """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE transition dipole moment
		trans = np.empty(info['trans'].shape[0], dtype=np.float64)
		for i in range(trans.size):
			trans[i] = np.linalg.norm(info['trans'][i, :])
		# plot results
		ax1.plot(np.arange(1, info['final_order']), \
				trans, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Transition dipole (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total transition dipole increments
		mbe = trans.copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.arange(1, info['final_order']), \
				np.abs(mbe), marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(tools.OUT+'/trans_dipole_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _osc_strength_plot(info, calc, exp):
		""" plot MBE oscillator strength """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE oscillator strength
		osc_strength = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			osc_strength[i] = (2./3.) * info['excitation'][i] * np.linalg.norm(info['trans'][i, :])**2
		# plot results
		ax1.plot(np.arange(1, info['final_order']), \
				osc_strength, marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Oscillator strength (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total oscillator strength increments
		mbe = osc_strength.copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.arange(1, info['final_order']), \
				np.abs(mbe), marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, info['final_order'] - 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(tools.OUT+'/osc_strength_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


