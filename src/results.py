#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, symm
from typing import Tuple, List, Any
try:
    import matplotlib
    PLT_FOUND = True
except (ImportError, OSError):
    pass
    PLT_FOUND = False
if PLT_FOUND:
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    try:
        import seaborn as sns
        SNS_FOUND = True
    except (ImportError, OSError):
        pass
        SNS_FOUND = False

from parallel import MPICls
from system import MolCls
from calculation import CalcCls
from expansion import ExpCls
from output import OUT, main_header
from tools import intervals, time_str


# results file
RES_FILE = OUT+'/pymbe.results'
# results parameters
DIVIDER = '{:^143}'.format('-'*137)
FILL = '{:^143}'.format('|'*137)


def main(mpi: MPICls, mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
        """
        this function handles all printing and plotting of results
        """
        # print header
        print(main_header())

        # print atom info
        if mol.atom:
            print(_atom(mol))

        # print summary
        print(_summary_prt(mpi, mol, calc, exp))

        # print timings
        print(_timings_prt(calc, exp))

        # print and plot results
        if calc.target_mbe == 'energy' :
            print(_energy_prt(calc, exp))
            if PLT_FOUND:
                _energies_plot(calc, exp)
        if calc.target_mbe == 'excitation':
            print(_excitation_prt(calc, exp))
            if PLT_FOUND:
                _excitation_plot(calc, exp)
        if calc.target_mbe == 'dipole' :
            print(_dipole_prt(mol, calc, exp))
            if PLT_FOUND:
                _dipole_plot(mol, calc, exp)
        if calc.target_mbe == 'trans':
            print(_trans_prt(mol, calc, exp))
            if PLT_FOUND:
                _trans_plot(mol, calc, exp)

        # plot number of determinants
        if PLT_FOUND:
            _max_ndets_plot(exp)


def _atom(mol: MolCls) -> str:
        """
        this function returns the molecular geometry
        """
        # print atom
        string: str = DIVIDER[:39]+'\n'
        string += '{:^43}\n'
        form: Tuple[Any] = ('geometry',)
        string += DIVIDER[:39]+'\n'
        molecule = gto.tostring(mol).split('\n')
        for i in range(len(molecule)):
            atom = molecule[i].split()
            for j in range(1, 4):
                atom[j] = float(atom[j]) # type: ignore
            string += '   {:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
            form += (*atom,) # type: ignore
        string += DIVIDER[:39]+'\n'
        return string.format(*form)


def _model(calc: CalcCls, x2c: bool) -> str:
        """
        this function returns the expansion model
        """
        string = '{:}'.format(calc.model['method'].upper())
        if x2c:
            string += ' (x2c)'
        return string


def _basis(mol: MolCls) -> str:
        """
        this function returns the basis
        """
        if isinstance(mol.basis, dict):

            for i, val in enumerate(mol.basis.items()):

                if i == 0:
                    basis = val[1]
                else:
                    basis += '/'+val[1]

            return basis

        else:

            return mol.basis


def _state(mol: MolCls, calc: CalcCls) -> str:
        """
        this function returns the state of interest
        """
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


def _ref(mol: MolCls, calc: CalcCls) -> str:
        """
        this function returns the reference function
        """
        if calc.ref['method'] == 'casci':
            return 'CASCI'
        else:
            if len(calc.ref['wfnsym']) == 1:
                return 'CASSCF'
            else:
                for i in range(len(set(calc.ref['wfnsym']))):
                    sym = symm.addons.irrep_id2name(mol.groupname, list(set(calc.ref['wfnsym']))[i])
                    num = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[i])
                    if i == 0:
                        syms = str(num)+'*'+sym
                    else:
                        syms += '/'+sym
                return 'CASSCF-'+syms


def _base(calc: CalcCls) -> str:
        """
        this function returns the base model
        """
        if calc.base['method'] is None:
            return 'none'
        else:
            return calc.base['method'].upper()


def _system(mol: MolCls) -> str:
        """
        this function returns the system size
        """
        return '{:} e in {:} o'.format(mol.nelectron - 2 * mol.ncore, mol.norb - mol.ncore)


def _hubbard(mol: MolCls) -> List[str]:
        """
        this function returns the hubbard model
        """
        hubbard = ['{:} x {:}'.format(mol.matrix[0], mol.matrix[1])]
        hubbard.append('{:} & {:}'.format(mol.u, mol.n))
        return hubbard


def _solver(calc: CalcCls) -> str:
        """
        this function returns the chosen fci solver
        """
        if calc.model['method'] != 'fci':
            return 'none'
        else:
            if calc.model['solver'] == 'pyscf_spin0':
                return 'PySCF (spin0)'
            elif calc.model['solver'] == 'pyscf_spin1':
                return 'PySCF (spin1)'
            else:
                raise NotImplementedError('unknown solver')


def _frozen(mol: MolCls) -> str:
        """
        this function returns the choice of frozen core
        """
        if mol.frozen:
            return 'true'
        else:
            return 'false'


def _active_space(calc: CalcCls) -> str:
        """
        this function returns the active space
        """
        string = '{:} e, {:} o'.format(calc.nelec[0] + calc.nelec[1], calc.ref_space.size)
        if calc.ref['active'] in ['avas', 'pios']:
            string += ' ({:})'.format(calc.ref['active'])
        return string


def _active_orbs(calc: CalcCls) -> str:
        """
        this function returns the orbitals of the active space
        """
        if calc.ref_space.size == 0:
            return 'none'

        # init string
        string = '['
        # divide ref_space into intervals
        ref_space_ints = [i for i in intervals(calc.ref_space)]

        for idx, i in enumerate(ref_space_ints):
            elms = '{:}-{:}'.format(i[0], i[1]) if len(i) > 1 else '{:}'.format(i[0])
            string += '{:},'.format(elms) if idx < len(ref_space_ints) - 1 else '{:}'.format(elms)
        string += ']'

        return string


def _orbs(calc: CalcCls) -> str:
        """
        this function returns the choice of orbitals
        """
        if calc.orbs['type'] == 'can':
            return 'canonical'
        elif calc.orbs['type'] == 'ccsd':
            return 'CCSD NOs'
        elif calc.orbs['type'] == 'ccsd(t)':
            return 'CCSD(T) NOs'
        elif calc.orbs['type'] == 'local':
            return 'pipek-mezey'
        else:
            raise NotImplementedError('unknown orbital basis')


def _mpi(mpi: MPICls) -> str:
        """
        this function returns the mpi information
        """
        return '{:} & {:}'.format(mpi.num_masters, mpi.global_size - mpi.num_masters)


def _point_group(mol: MolCls) -> str:
        """
        this function returns the molecular point group symmetry
        """
        return mol.topgroup


def _comp_point_group(mol: MolCls) -> str:
        """
        this function returns the computational point group symmetry
        """
        return mol.groupname


def _symm(mol: MolCls, calc: CalcCls) -> str:
        """
        this function returns the symmetry of the wavefunction in the
        computational point group
        """
        if calc.model['method'] == 'fci':
            if mol.atom:
                string = symm.addons.irrep_id2name(mol.groupname, calc.state['wfnsym'])+'('+mol.groupname+')'
                if calc.extra['pi_prune']:
                    string += ' (pi)'
                return string
            else:
                return 'A(C1)'
        else:
            return 'unknown'


def _energy(calc: CalcCls, exp: ExpCls) -> np.ndarray:
        """
        this function returns the final total energy
        """
        e_tot = np.copy(exp.prop['energy']['tot'])
        e_tot += calc.prop['hf']['energy']
        e_tot += calc.prop['base']['energy']
        e_tot += calc.prop['ref']['energy']

        return e_tot


def _excitation(calc: CalcCls, exp: ExpCls) -> np.ndarray:
        """
        this function returns the final excitation energy
        """
        exc_tot = np.copy(exp.prop['excitation']['tot'])
        exc_tot += calc.prop['ref']['excitation']

        return exc_tot


def _dipole(mol: MolCls, calc: CalcCls, exp: ExpCls) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the final molecular dipole moment
        """
        # nuclear dipole moment
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nuc_dipole = np.einsum('i,ix->x', charges, coords)

        dipole_tot = np.copy(exp.prop['dipole']['tot'])
        dipole_tot += calc.prop['hf']['dipole']
        dipole_tot += calc.prop['base']['dipole']
        dipole_tot += calc.prop['ref']['dipole']

        return dipole_tot, nuc_dipole


def _trans(mol: MolCls, calc: CalcCls, exp: ExpCls) -> np.ndarray:
        """
        this function returns the final molecular transition dipole moment
        """
        trans_tot = np.copy(exp.prop['trans']['tot'])
        trans_tot += calc.prop['ref']['trans']

        return trans_tot


def _time(exp: ExpCls, comp: str, idx: int) -> str:
        """
        this function returns the final timings in (HHH : MM : SS) format
        """
        # init time
        if comp in ['mbe', 'purge']:
            time = exp.time[comp][idx]
        elif comp == 'sum':
            time = exp.time['mbe'][idx] + exp.time['purge'][idx]
        elif comp in ['tot_mbe', 'tot_purge']:
            time = np.sum(exp.time[comp[4:]])
        elif comp == 'tot_sum':
            time = np.sum(exp.time['mbe']) + np.sum(exp.time['purge'])
        return time_str(time)


def _summary_prt(mpi: MPICls, mol: MolCls, calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the summary table
        """
        if calc.target_mbe == 'energy':
            hf_prop = calc.prop['hf']['energy']
            base_prop = calc.prop['hf']['energy']+calc.prop['base']['energy']
            mbe_tot_prop = _energy(calc, exp)[-1].item()
        elif calc.target_mbe == 'dipole':
            dipole, nuc_dipole = _dipole(mol, calc, exp)
            hf_prop = np.linalg.norm(nuc_dipole - calc.prop['hf']['dipole'])
            base_prop = np.linalg.norm(nuc_dipole - (calc.prop['hf']['dipole'] + calc.prop['base']['dipole']))
            mbe_tot_prop = np.linalg.norm(nuc_dipole - dipole[-1, :])
        elif calc.target_mbe == 'excitation':
            hf_prop = 0.
            base_prop = 0.
            mbe_tot_prop = _excitation(calc, exp)[-1].item()
        else:
            hf_prop = 0.
            base_prop = 0.
            mbe_tot_prop = np.linalg.norm(_trans(mol, calc, exp)[-1, :])

        string: str = DIVIDER+'\n'
        string += '{:14}{:21}{:12}{:1}{:12}{:21}{:11}{:1}{:13}{:}\n'
        form: Tuple[Any, ...] = ('','molecular information','','|','', \
                    'expansion information','','|','','calculation information',)
        string += DIVIDER+'\n'

        if mol.atom:

            string += '{:9}{:18}{:2}{:1}{:2}{:<14s}{:1}{:1}{:7}{:15}{:2}{:1}{:2}' \
                        '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
            form += ('','basis set','','=','',_basis(mol), \
                        '','|','','expansion model','','=','',_model(calc, mol.x2c), \
                        '','|','','mpi masters & slaves','','=','',_mpi(mpi),)

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:23}{:1}{:1}{:2}{:.6f}\n'
            form += ('','frozen core','','=','',_frozen(mol), \
                        '','|','','reference','','=','',_ref(mol, calc), \
                        '','|','','Hartree-Fock '+calc.target_mbe,'','=','',hf_prop,)

        else:

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
            form += ('','hubbard matrix','','=','',_hubbard(mol)[0], \
                        '','|','','expansion model','','=','',_model(calc, mol.x2c), \
                        '','|','','mpi masters & slaves','','=','',_mpi(mpi),)

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:.6f}\n'
            form += ('','hubbard U/t & n','','=','',_hubbard(mol)[1], \
                        '','|','','reference','','=','',_ref(mol, calc), \
                        '','|','','Hartree-Fock '+calc.target_mbe,'','=','',hf_prop,)

        string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:.6f}\n'
        form += ('','system size','','=','',_system(mol), \
                    '','|','','reference space','','=','',_active_space(calc), \
                    '','|','','base model '+calc.target_mbe,'','=','',base_prop,)

        string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:.6f}\n'
        form += ('','state (mult.)','','=','',_state(mol, calc), \
                    '','|','','reference orbs.','','=','',_active_orbs(calc), \
                    '','|','','MBE total '+calc.target_mbe,'','=','',mbe_tot_prop,)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','orbitals','','=','',_orbs(calc), \
                    '','|','','base model','','=','',_base(calc), \
                    '','|','','total time','','=','',_time(exp, 'tot_sum', -1),)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','full point group','','=','',_point_group(mol), \
                    '','|','','','','','','', \
                    '','|','','wave funct. symmetry','','=','',_symm(mol, calc),)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','comp. point group','','=','',_comp_point_group(mol), \
                    '','|','','','','','','', \
                    '','|','','','','','','',)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','FCI solver','','=','',_solver(calc), \
                    '','|','','','','','','', \
                    '','|','','','','','','',)

        string += DIVIDER+'\n'+FILL+'\n'+DIVIDER+'\n'

        return string.format(*form)


def _timings_prt(calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the timings table
        """
        string: str = DIVIDER[:112]+'\n'
        string += '{:^112}\n'
        form: Tuple[Any, ...] = ('MBE-{:} timings'.format(calc.model['method'].upper()),)

        string += DIVIDER[:112]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:3}{:8}{:1}{:6}{:7}{:6}' \
                  '{:1}{:8}{:3}{:8}{:1}{:4}{:12s}{:4}{:1}{:5}{:4}\n'
        form += ('','MBE order','','|','','MBE','',
                 '|','','purging','','|','','sum','',\
                 '|','','calculations','','|','','in %',)

        string += DIVIDER[:112]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            calc_i = exp.n_tuples['inc'][i]
            rel_i = exp.n_tuples['inc'][i] / exp.n_tuples['theo'][i] * 100.
            calc_tot = sum(exp.n_tuples['inc'][:i+1])
            rel_tot = calc_tot / sum(exp.n_tuples['theo'][:i+1]) * 100.
            string += '{:7}{:>4d}{:6}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}' \
                      '{:1}{:2}{:>15s}{:2}{:1}{:5}{:>10d}{:5}{:1}{:4}{:6.2f}\n'
            form += ('',j, \
                     '','|','',_time(exp, 'mbe', i), \
                     '','|','',_time(exp, 'purge', i), \
                     '','|','',_time(exp, 'sum', i), \
                     '','|','',calc_i,'','|','',rel_i,)

        string += DIVIDER[:112]+'\n'
        string += '{:8}{:5s}{:4}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}' \
                  '{:1}{:2}{:>15s}{:2}{:1}{:5}{:>10d}{:5}{:1}{:4}{:6.2f}\n'
        form += ('','total', \
                 '','|','',_time(exp, 'tot_mbe', -1), \
                 '','|','',_time(exp, 'tot_purge', -1), \
                 '','|','',_time(exp, 'tot_sum', -1), \
                 '','|','',calc_tot,'','|','',rel_tot,)

        string += DIVIDER[:112]+'\n'

        return string.format(*form)


def _energy_prt(calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the energies table
        """
        string: str = DIVIDER[:66]+'\n'
        string_in = 'MBE-{:} energy (root = {:})'.format(calc.model['method'].upper(), calc.state['root'])
        string += '{:^66}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:66]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:12}{:5}{:1}{:4}{:}\n'
        form += ('','MBE order','','|','','total energy','','|','','correlation energy',)

        string += DIVIDER[:66]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:5}{:>11.6f}{:6}{:1}{:6}{:>12.5e}\n'
        form += ('','ref','','|','',calc.prop['hf']['energy'] + calc.prop['ref']['energy'], \
                    '','|','',calc.prop['ref']['energy'],)

        string += DIVIDER[:66]+'\n'
        energy = _energy(calc, exp)

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:5}{:>11.6f}{:6}{:1}{:6}{:>12.5e}\n'
            form += ('',j, \
                        '','|','',energy[i].item(), \
                        '','|','',energy[i].item() - calc.prop['hf']['energy'],)

        string += DIVIDER[:66]+'\n'

        return string.format(*form)


def _energies_plot(calc: CalcCls, exp: ExpCls) -> None:
        """
        this function plots the energies
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                _energy(calc, exp), marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
                linestyle='-', label='state {:}'.format(calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Energy (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1, frameon=False)

        # save plot
        plt.savefig(OUT+'/energy_state_{:}.pdf'. \
                    format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _excitation_prt(calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the excitation energies table
        """
        string: str = DIVIDER[:43]+'\n'
        string_in = 'MBE excitation energy (root = '+str(calc.state['root'])+')'
        string += '{:^46}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:43]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','excitation energy',)

        string += DIVIDER[:43]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:7}{:>.5e}\n'
        form += ('','ref','','|','',calc.prop['ref']['excitation'],)

        string += DIVIDER[:43]+'\n'
        excitation = _excitation(calc, exp)

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:7}{:>.5e}\n'
            form += ('',j,'','|','',excitation[i].item(),)

        string += DIVIDER[:43]+'\n'

        return string.format(*form)


def _excitation_plot(calc: CalcCls, exp: ExpCls) -> None:
        """
        this function plots the excitation energies
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                _excitation(calc, exp), marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Excitation energy (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1, frameon=False)

        # save plot
        plt.savefig(OUT+'/excitation_states_{:}_{:}.pdf'. \
                    format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _dipole_prt(mol: MolCls, calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the dipole moments table
        """
        string: str = DIVIDER[:82]+'\n'
        string_in = 'MBE dipole moment (root = {:}) - gauge origin: ({:.3f}, {:.3f}, {:.3f})'.\
                        format(calc.state['root'], *mol.gauge_origin)
        string += '{:^82}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:82]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)

        string += DIVIDER[:82]+'\n'

        dipole_ref: np.ndarray = calc.prop['hf']['dipole'] + \
                                    calc.prop['base']['dipole'] + \
                                    calc.prop['ref']['dipole']
        dipole, nuc_dipole = _dipole(mol, calc, exp)
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',nuc_dipole[0] - dipole_ref[0], \
                    '',nuc_dipole[1] - dipole_ref[1], \
                    '',nuc_dipole[2] - dipole_ref[2], \
                    '','|','',np.linalg.norm(nuc_dipole - dipole_ref),)

        string += DIVIDER[:82]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
            form += ('',j, \
                        '','|','',nuc_dipole[0] - dipole[i, 0], \
                        '',nuc_dipole[1] - dipole[i, 1], \
                        '',nuc_dipole[2] - dipole[i, 2], \
                        '','|','',np.linalg.norm(nuc_dipole - dipole[i, :]),)

        string += DIVIDER[:82]+'\n'

        return string.format(*form)


def _dipole_plot(mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
        """
        this function plots the dipole moments
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE dipole moment
        dipole, nuc_dipole = _dipole(mol, calc, exp)
        dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
        for i in range(dipole.shape[0]):
            dipole_arr[i] = np.linalg.norm(nuc_dipole - dipole[i, :])

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                dipole_arr, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
                linestyle='-', label='state {:}'.format(calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Dipole moment (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1, frameon=False)

        # save plot
        plt.savefig(OUT+'/dipole_state_{:}.pdf'. \
                    format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _trans_prt(mol: MolCls, calc: CalcCls, exp: ExpCls) -> str:
        """
        this function returns the transition dipole moments and oscillator strengths table
        """
        string: str = DIVIDER[:82]+'\n'
        string_in = 'MBE trans. dipole moment (roots 0 > {:}) - gauge origin: ({:.3f}, {:.3f}, {:.3f})'.\
                        format(calc.state['root'], *mol.gauge_origin)
        string += '{:^84}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:82]+'\n'

        trans = _trans(mol, calc, exp)

        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)

        trans_ref: np.ndarray = calc.prop['ref']['trans']
        string += DIVIDER[:82]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',trans_ref[0], '', trans_ref[1], '', trans_ref[2], \
                    '','|','',np.linalg.norm(trans_ref[:]),)

        string += DIVIDER[:82]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
            form += ('',j, \
                        '','|','',trans[i, 0], \
                        '',trans[i, 1], \
                        '',trans[i, 2], \
                        '','|','',np.linalg.norm(trans[i, :]),)

        string += DIVIDER[:82]+'\n'

        return string.format(*form)


def _trans_plot(mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
        """
        this function plots the transition dipole moments
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE transition dipole moment
        trans = _trans(mol, calc, exp)
        trans_arr = np.empty(trans.shape[0], dtype=np.float64)
        for i in range(trans.shape[0]):
            trans_arr[i] = np.linalg.norm(trans[i, :])

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                trans_arr, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Transition dipole (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1, frameon=False)

        # save plot
        plt.savefig(OUT+'/trans_dipole_states_{:}_{:}.pdf'. \
                    format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _max_ndets_plot(exp: ExpCls) -> None:
        """
        this function plots the max number of determinants
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.semilogy(np.arange(exp.min_order, exp.final_order+1), \
                    exp.max_ndets, marker='x', linewidth=2, mew=1, color='red', linestyle='-')

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Number of determinants')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # save plot
        plt.savefig(OUT+'/max_ndets.pdf', bbox_inches = 'tight', dpi=1000)


