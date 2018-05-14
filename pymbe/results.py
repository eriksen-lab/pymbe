#!/usr/bin/env python
# -*- coding: utf-8 -*

""" results.py: summary and plotting module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
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
except ImportError:
	sys.stderr.write('\nImportError : seaborn module not found\n\n')


# results parameters
OUT = os.getcwd()+'/output'
DIVIDER = '{0:^143}'.format('-'*137)
FILL = '{0:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
		""" printing and plotting of results """
		# convert final results to numpy arrays
		exp.property['energy']['tot'] = np.asarray(exp.property['energy']['tot'])
		if calc.prop['EXCITATION']:
			exp.property['excitation']['tot'] = np.asarray(exp.property['excitation']['tot'])
		if calc.prop['DIPOLE']:
			exp.property['dipole']['tot'] = np.asarray(exp.property['dipole']['tot'])
		# setup
		info = {}
		info['model_type'], info['basis'], info['mult'], info['ref'], info['base'], info['prot'], \
			info['system'], info['frozen'], info['active'], info['occ'], info['virt'], \
			info['mpi'], info['thres'], info['symm'], \
			info['e_final'], info['e_exc_final'], \
			info['dipole_final'], info['dipole_hf'] = _setup(mpi, mol, calc, exp)
		# results
		_table(info, mol, calc, exp)
		# plot
		_plot(info, calc, exp)


def _setup(mpi, mol, calc, exp):
		""" init parameters """
		model_type = _model_type(calc)
		basis = _basis(mol)
		mult = _mult(mol)
		ref = _ref(mol, calc)
		base = _base(calc)
		prot = _prot(calc)
		system = _system(mol, calc)
		frozen = _frozen(mol)
		active = _active(calc)
		occ, virt = _orbs(calc)
		mpi = _mpi(mpi, calc)
		thres = _thres(calc)
		symm = _symm(mol, calc)
		e_final = _e_final(calc, exp)
		if calc.prop['EXCITATION']:
			e_exc_final = _e_exc_final(calc, exp)
		else:
			e_exc_final = None
		if calc.prop['DIPOLE']:
			dipole_final, dipole_hf = _dipole_final(mol, calc, exp)
		else:
			dipole_final = dipole_hf = None
		return model_type, basis, mult, ref, base, prot, system, frozen, \
				active, occ, virt, mpi, thres, symm, e_final, \
				e_exc_final, dipole_final, dipole_hf


def _table(info, mol, calc, exp):
		""" print results """
		# write results to results.out
		with open(OUT+'/results.out','a') as f:
			with contextlib.redirect_stdout(f):
				_summary_prt(info, calc, exp)
				_timings_prt(exp)
				_energy_prt(info, calc, exp)
				if calc.prop['EXCITATION']:
					_excitation_prt(info, calc, exp)
				if calc.prop['DIPOLE']:
					_dipole_prt(info, mol, calc, exp)
	

def _plot(info, calc, exp):
		""" plot results """
		# plot MBE energies
		_energies_plot(info, calc, exp)
		# plot maximal increments
		_max_inc_plot(calc, exp)
		# plot MBE excitation energy
		if calc.prop['EXCITATION']:
			_excitation_plot(info, calc, exp)
		# plot MBE dipole moment
		if calc.prop['DIPOLE']:
			_dipole_plot(info, calc, exp)


def _summary_prt(info, calc, exp):
		""" summary table """
		print(DIVIDER)
		print('{0:14}{1:21}{2:12}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'. \
				format('','molecular information','','|','', \
					'expansion information','','|','','calculation information'))
		print(DIVIDER)
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
					format('','basis set','','=','',info['basis'], \
						'','|','','model / type','','=','',info['model_type'], \
						'','|','','mpi masters / slaves','','=','',info['mpi']))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','spin multiplicity','','=','',info['mult'], \
						'','|','','ref. function','','=','',info['ref'], \
						'','|','','Hartree-Fock energy','','=','',calc.property['energy']['hf']))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','system size','','=','',info['system'], \
						'','|','','cas size','','=','',info['active'], \
						'','|','','base model energy','','=','', \
						calc.property['energy']['hf']+calc.property['energy']['base']))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','frozen core','','=','',info['frozen'], \
						'','|','','base model','','=','',info['base'], \
						'','|','','MBE total energy','','=','',info['e_final'][-1]))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:}{22:<2s}{23:}{24:<2s}{25:}{26:<1s}'.\
					format('','occupied orbs','','=','',info['occ'], \
						'','|','','screen. prot.','','=','',info['prot'], \
						'','|','','total time','','=','', \
						_time(exp, 'total', exp.order-1)[0],'h', \
						_time(exp, 'total', exp.order-1)[1],'m', \
						_time(exp, 'total', exp.order-1)[2],'s'))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
					format('','virtual orbs','','=','',info['virt'], \
						'','|','','screen. thres.','','=','',info['thres'], \
						'','|','','wave funct. symmetry','','=','',info['symm']))
		print(DIVIDER)
		print(FILL)
		print(DIVIDER+'\n')


def _timings_prt(exp):
		""" timings """
		print(DIVIDER[:98])
		print('{0:^98}'.format('MBE timings'))
		print(DIVIDER[:98])
		print('{0:6}{1:9}{2:2}{3:1}{4:6}{5:47}{6:7}{7:1}{8:4}{9:}'. \
				format('','MBE order','','|','','time (HHH : MM : SS) -- MBE / screening - total', \
						'','|','','calculations'))
		print(DIVIDER[:98])
		for i in range(exp.property['energy']['tot'].size):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:3d}{6:^3}{7:2d}{8:^3}{9:2d}{10:^5}{11:3d}'
				'{12:^3}{13:2d}{14:^3}{15:2d}{16:^5}{17:3d}{18:^3}{19:2d}{20:^3}{21:2d}'
				'{22:7}{23:1}{24:6}{25:<9d}'. \
					format('',i+exp.start_order, \
						'','|','',_time(exp, 'mbe', i)[0],':', \
						_time(exp, 'mbe', i)[1],':', \
						_time(exp, 'mbe', i)[2], \
						'/',_time(exp, 'screen', i)[0],':', \
   						_time(exp, 'screen', i)[1],':', \
   						_time(exp, 'screen', i)[2], \
						'-',_time(exp, 'total', i)[0],':', \
 						_time(exp, 'total', i)[1],':', \
 						_time(exp, 'total', i)[2], \
						'','|','',exp.tuples[i].shape[0]))
		print(DIVIDER[:98]+'\n')


def _energy_prt(info, calc, exp):
		""" ground state energy """
		print(DIVIDER[:66])
		print('{0:^66}'.format('MBE ground state energy'))
		print(DIVIDER[:66])
		print('{0:6}{1:9}{2:2}{3:1}{4:5}{5:12}{6:5}{7:1}{8:4}{9:}'. \
				format('','MBE order','','|','','total energy','','|','','correlation energy'))
		print(DIVIDER[:66])
		for i in range(exp.property['energy']['tot'].size):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:>11.6f}{6:6}{7:1}{8:7}{9:9.4e}'. \
					format('',i+exp.start_order, \
						'','|','',info['e_final'][i], \
						'','|','',info['e_final'][i] - calc.property['energy']['hf']))
		print(DIVIDER[:66]+'\n')


def _excitation_prt(info, calc, exp):
		""" excitation energy """
		print(DIVIDER[:66])
		string = 'MBE excited state energy (root = {0:})'.format(calc.state['ROOT'])
		print('{0:^66}'.format(string))
		print(DIVIDER[:66])
		print('{0:6}{1:9}{2:2}{3:1}{4:5}{5:12}{6:5}{7:1}{8:5}{9:}'. \
				format('','MBE order','','|','','total energy','','|','','excitation energy'))
		print(DIVIDER[:66])
		for i in range(exp.property['energy']['tot'].size):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:>11.6f}{6:6}{7:1}{8:8}{9:9.4e}'. \
					format('',i+exp.start_order, \
						'','|','',info['e_final'][i] + info['e_exc_final'][i], \
						'','|','',info['e_exc_final'][i]))
		print(DIVIDER[:66]+'\n')


def _dipole_prt(info, mol, calc, exp):
		""" dipole moment """
		print(DIVIDER[:110])
		print('{0:^110}'.format('MBE ground state dipole moment'))
		print(DIVIDER[:110])
		print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:25}{6:9}{7:1}{8:5}{9:13}{10:5}{11:1}{12:4}{13:}'. \
				format('','MBE order','','|','','dipole components (x,y,z)', \
						'','|','','dipole moment','','|','','correlation dipole'))
		print(DIVIDER[:110])
		for i in range(exp.property['energy']['tot'].size):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:4}{5:9.6f}{6:^3}{7:9.6f}{8:^3}{9:9.6f}'
				'{10:5}{11:1}{12:7}{13:9.6f}{14:7}{15:1}{16:8}{17:9.6f}'. \
					format('',i+exp.start_order, \
						'','|','',info['dipole_final'][i, 0], \
						'',info['dipole_final'][i, 1], \
						'',info['dipole_final'][i, 2], \
						'','|','',np.sqrt(np.sum(info['dipole_final'][i, :]**2)), \
						'','|','',np.sqrt(np.sum(info['dipole_final'][i, :]**2)) \
									- np.sqrt(np.sum(info['dipole_hf']**2))))
		print(DIVIDER[:110]+'\n')


def _model_type(calc):
		""" model / type print """
		return '{0:} / {1:}'.format(calc.model['METHOD'], calc.model['TYPE'].lower())


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


def _mult(mol):
		""" mult print """
		if mol.spin == 0:
			return 'singlet'
		elif mol.spin == 1:
			return 'doublet'
		elif mol.spin == 2:
			return 'triplet'
		elif mol.spin == 3:
			return 'quartet'
		elif mol.spin == 4:
			return 'quintet'
		else:
			return '{0:}'.format(mol.spin+1)


def _ref(mol, calc):
		""" ref print """
		if calc.ref['METHOD'] == 'HF':
			if mol.spin == 0:
				return 'RHF'
			else:
				return 'ROHF'
		else:
			return calc.ref['METHOD']


def _base(calc):
		""" base print """
		if calc.base['METHOD'] is None:
			return 'none'
		else:
			return calc.base['METHOD']


def _prot(calc):
		""" protocol print """
		prot = calc.prot['SCHEME'].lower()
		if calc.prot['ENERGY_ONLY']:
			prot += ' (energy)'
		else:
			prot += ' (all props.)'
		return prot


def _system(mol, calc):
		""" system size print """
		return '{0:} e / {1:} o'.format(mol.nelectron - 2*mol.ncore, len(calc.ref_space) + len(calc.exp_space))


def _frozen(mol):
		""" frozen core print """
		if mol.frozen:
			return 'true'
		else:
			return 'false'


def _active(calc):
		""" active space print """
		if calc.ref['METHOD'] == 'HF':
			return 'none'
		else:
			return '{0:} e / {1:} o'.format(calc.ne_act[0] + calc.ne_act[1], calc.no_act)


def _orbs(calc):
		""" orbital print """
		if calc.orbs['OCC'] == 'CAN':
			occ = 'canonical'
		elif calc.orbs['OCC'] == 'CISD':
			occ = 'CISD natural'
		elif calc.orbs['OCC'] == 'CCSD':
			occ = 'CCSD natural'
		elif calc.orbs['OCC'] == 'SCI':
			occ = 'SCI natural'
		elif calc.orbs['OCC'] == 'PM':
			occ = 'pipek-mezey'
		elif calc.orbs['OCC'] == 'FB':
			occ = 'foster-boys'
		elif calc.orbs['OCC'] == 'IBO-1':
			occ = 'intrin. bond'
		elif calc.orbs['OCC'] == 'IBO-2':
			occ = 'intrin. bond'
		if calc.orbs['VIRT'] == 'CAN':
			virt = 'canonical'
		elif calc.orbs['VIRT'] == 'CISD':
			virt = 'CISD natural'
		elif calc.orbs['VIRT'] == 'CCSD':
			virt = 'CCSD natural'
		elif calc.orbs['VIRT'] == 'SCI':
			virt = 'SCI natural'
		elif calc.orbs['VIRT'] == 'PM':
			virt = 'pipek-mezey'
		elif calc.orbs['VIRT'] == 'FB':
			virt = 'foster-boys'
		return occ, virt


def _mpi(mpi, calc):
		""" mpi print """
		return '{0:} / {1:}'.format(calc.mpi['MASTERS'], mpi.global_size - calc.mpi['MASTERS'])


def _thres(calc):
		""" threshold print """
		return '{0:.0e} / {1:<.1f}'.format(calc.thres['INIT'], calc.thres['RELAX'])


def _symm(mol, calc):
		""" symmetry print """
		if calc.model['METHOD'] in ['SCI','FCI']:
			return symm.addons.irrep_id2name(mol.symmetry, calc.state['WFNSYM'])+' ('+mol.symmetry+')'
		else:
			return 'unknown'


def _e_final(calc, exp):
		""" final ground state energy """
		return exp.property['energy']['tot'] \
				+ calc.property['energy']['hf'] + calc.property['energy']['base'] \
				+ (calc.property['energy']['ref'] - calc.property['energy']['ref_base'])


def _e_exc_final(calc, exp):
		""" final excitation energy """
		return exp.property['excitation']['tot'] + calc.property['excitation']['ref']


def _dipole_final(mol, calc, exp):
		""" final ground state molecular dipole moment """
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipole = np.einsum('i,ix->x', charges, coords)
		# molecular dipole moment
		dipole = nuc_dipole \
					- (exp.property['dipole']['tot'] \
						+ calc.property['dipole']['hf'] \
						+ calc.property['dipole']['ref'])
		dipole_hf = nuc_dipole - calc.property['dipole']['hf']
		return dipole, dipole_hf


def _time(exp, comp, idx):
		""" convert time to (HHH : MM : SS) format """
		if comp != 'total':
			hours = int(exp.time[comp][idx]//3600)
			minutes = int((exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600.)//60)
			seconds = int(exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600. \
							- ((exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600.)//60)*60.)
		else:
			hours = int(np.sum(exp.time[comp][:idx+1])//3600)
			minutes = int((np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600.)//60)
			seconds = int(np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600. \
							- ((np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600.)//60)*60.)
		return hours, minutes, seconds


def _energies_plot(info, calc, exp):
		""" plot MBE energies """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy increments
		mbe_gs = info['e_final'] - calc.property['energy']['hf']
		mbe_gs[1:] = np.diff(mbe_gs)
		if calc.prop['EXCITATION']:
			mbe_ex = info['e_final'] + info['e_exc_final'] - calc.property['energy']['hf']
			mbe_ex[1:] = np.diff(mbe_ex)
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				info['e_final'], marker='x', linewidth=2, color='green', \
				linestyle='-', label='ground state')
		if calc.prop['EXCITATION']:
			ax1.plot(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
					info['e_final'] + info['e_exc_final'], marker='x', linewidth=2, color='blue', \
					linestyle='-', label='excited state {0:}'.format(calc.state['ROOT']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				np.abs(mbe_gs), marker='x', linewidth=2, color='green', \
				linestyle='-', label='ground state')
		if calc.prop['EXCITATION']:
			ax2.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
					np.abs(mbe_ex), marker='x', linewidth=2, color='blue', \
					linestyle='-', label='excited state {0:}'.format(calc.state['ROOT']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		sns.despine()
		# set legends
		ax2.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/energies.pdf', bbox_inches = 'tight', dpi=1000)


def _max_inc_plot(calc, exp):
		""" plot maximal increments """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# array of max increments
		e_max = np.empty_like(exp.property['energy']['tot'])
		if calc.prop['EXCITATION']:
			e_exc_max = np.empty_like(e_max)
		if calc.prop['DIPOLE']:
			dip_max = np.empty_like(e_max)
		for i in range(exp.property['energy']['tot'].size):
			max_idx = np.argmax(np.abs(exp.property['energy']['inc'][i]))
			e_max[i] = np.abs(exp.property['energy']['inc'][i][max_idx])
			if calc.prop['EXCITATION']:
				max_idx = np.argmax(np.abs(exp.property['excitation']['inc'][i]))
				e_exc_max[i] = np.abs(exp.property['excitation']['inc'][i][max_idx])
			if calc.prop['DIPOLE']:
				max_idx = np.unravel_index(np.argmax(np.abs(exp.property['dipole']['inc'][i]), axis=None), \
											exp.property['dipole']['inc'][i].shape)
				dip_max[i] = np.abs(exp.property['dipole']['inc'][i][max_idx])
		# plot results
		ax.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				e_max, marker='x', linewidth=2, color=sns.xkcd_rgb['salmon'], \
				linestyle='-', label='ground state energy')
		if calc.prop['EXCITATION']:
			ax.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
					e_exc_max, marker='x', linewidth=2, color=sns.xkcd_rgb['royal blue'], \
					linestyle='-', label='excitation energy for state {0:}'.format(calc.state['ROOT']))
		if calc.prop['DIPOLE']:
			ax.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
					dip_max, marker='x', linewidth=2, color=sns.xkcd_rgb['kelly green'], \
					linestyle='-', label='ground state dipole moment')
		# set x limits
		ax.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax.xaxis.grid(False)
		# set labels
		ax.set_xlabel('Expansion order')
		ax.set_ylabel('Max. increments (in au)')
		# force integer ticks on x-axis
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
		# despine
		sns.despine()
		# set legends
		ax.legend(loc=1)
		# tight layout
		plt.tight_layout()
		# save plot
		plt.savefig(OUT+'/max_inc.pdf', bbox_inches = 'tight', dpi=1000)


def _excitation_plot(info, calc, exp):
		""" plot MBE excitation energy """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE excitation energy increments
		mbe = info['e_exc_final'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				info['e_exc_final'], marker='x', linewidth=2, color=sns.xkcd_rgb['salmon'], \
				linestyle='-', label='excited state {0:}'.format(calc.state['ROOT']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Excitation energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.4f'))
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				np.abs(mbe), marker='x', linewidth=2, color=sns.xkcd_rgb['salmon'], \
				linestyle='-', label='excited state {0:}'.format(calc.state['ROOT']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		sns.despine()
		# set legends
		ax2.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/excitation.pdf', bbox_inches = 'tight', dpi=1000)


def _dipole_plot(info, calc, exp):
		""" plot MBE dipole moment """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# dipole length
		dipole = np.empty_like(exp.property['energy']['tot'])
		for i in range(dipole.size):
			dipole[i] = np.sqrt(np.sum(info['dipole_final'][i, :]**2))
		# array of MBE dipole increments
		mbe = np.empty_like(exp.property['energy']['tot'])
		for i in range(mbe.size):
			mbe[i] = np.sqrt(np.sum(info['dipole_final'][i, :]**2)) - np.sqrt(np.sum(info['dipole_hf']**2))
		mbe[1:] = np.diff(mbe)
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				dipole, marker='x', linewidth=2, color='red', \
				linestyle='-', label='ground state')
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Dipole moment (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, exp.property['energy']['tot'].size+exp.start_order))), \
				np.abs(mbe), marker='x', linewidth=2, color='red', \
				linestyle='-', label='ground state')
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		sns.despine()
		# set legends
		ax2.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/dipole.pdf', bbox_inches = 'tight', dpi=1000)


