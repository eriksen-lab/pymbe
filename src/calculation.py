#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
calculation module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import re
import sys
import os
import numpy as np
from ast import literal_eval
from pyscf import symm, scf
from typing import Dict, List, Tuple, Union, Any

from tools import assertion


# attributes
ATTR = ['model', 'hf_ref', 'base', 'orbs', 'target', 'thres', 'mpi', 'extra', 'misc', 'ref', 'state']


class CalcCls:
        """
        this class contains the pymbe calculation attributes
        """
        def __init__(self, ncore: int, nelectron: int, symmetry: Union[bool, str]) -> None:
                """
                init calculation attributes
                """
                # set defaults
                self.model: Dict[str, Any] = {'method': 'fci', 'cc_backend': 'pyscf', 'solver': 'pyscf_spin0', 'hf_guess': True}
                self.hf_ref: Dict[str, Any] = {'symmetry': None, 'irrep_nelec': {}, \
                                               'init_guess': 'minao', 'newton': False}
                self.target: Dict[str, bool] = {'energy': False, 'excitation': False, 'dipole': False, 'trans': False}
                self.ref: Dict[str, Any] = {'method': 'casci', 'hf_guess': True, 'active': 'manual', \
                                            'select': [i for i in range(ncore, nelectron // 2)], 'weights': [1.], \
                                            'wfnsym': [symm.addons.irrep_id2name(symmetry, 0) if symmetry else 0], \
                                            'pi-atoms': [], 'ao-labels': []}
                self.base: Dict[str, Union[None, str]] = {'method': None}
                self.state: Dict[str, Any] = {'wfnsym': symm.addons.irrep_id2name(symmetry, 0) if symmetry else 0, 'root': 0}
                self.extra: Dict[str, bool] = {'pi_prune': False}
                self.thres: Dict[str, Union[int, float]] = {'start': 4, 'perc': .9}
                self.misc: Dict[str, Any] = {'order': None, 'rst': True, 'rst_freq': int(1e6), 'purge': True}
                self.orbs: Dict[str, str] = {'type': 'can'}
                self.mpi: Dict[str, int] = {}
                self.prop: Dict[str, Dict[str, Union[float, np.ndarray]]] = {'hf': {}, 'base': {}, 'ref': {}}
                # init attributes
                self.restart: bool = False
                self.target_mbe: str = ''
                self.hf: scf.RHF = None
                self.occup: np.ndarray = None
                self.orbsym: np.ndarray = None
                self.mo_coeff: np.ndarray = None
                self.nelec: Tuple[int, ...] = ()
                self.ref_space: np.ndarray = None
                self.mut_info: np.ndarray = None


def set_calc(calc: CalcCls) -> CalcCls:
        """
        this function sets calculation and mpi attributes from input file
        """
        # read input file
        try:
            with open(os.getcwd()+'/input') as f:
                content = f.readlines()
                for i in range(len(content)):
                    if content[i].strip():
                        if content[i].split()[0][0] == '#':
                            continue
                        else:
                            attr = re.split('=',content[i])[0].strip()
                            if attr in ATTR:
                                try:
                                    inp = literal_eval(re.split('=',content[i])[1].strip())
                                except ValueError:
                                    raise ValueError('wrong input -- error in reading in {:} dictionary'.format(attr))
                                # make keys uniformly lower-case
                                keys = [key.lower() for key in inp.keys()]
                                # make string values lower-case as well
                                vals = [val.lower() if isinstance(val, str) else val for val in inp.values()]
                                # recast wfnsym as standard symbol
                                if 'wfnsym' in inp.keys():
                                    if attr == 'state':
                                        inp['wfnsym'] = symm.addons.std_symb(inp['wfnsym'])
                                    elif attr == 'ref':
                                        if not isinstance(inp['wfnsym'], list):
                                            inp['wfnsym'] = list(inp['wfnsym'])
                                        inp['wfnsym'] = [symm.addons.std_symb(sym) for sym in inp['wfnsym']]
                                # update calc attribute
                                setattr(calc, attr, {**getattr(calc, attr), **inp})
        except IOError:
            sys.stderr.write('\nIOError : input file not found\n\n')
            raise
        return calc


def sanity_check(calc: CalcCls, spin: int, atom: Union[List[str], str], \
                symmetry: Union[bool, str]) -> None:
        """
        this function performs sanity checks of calc and mpi attributes
        """
        # expansion model
        assertion(isinstance(calc.model['method'], str), \
                        'input electronic structure method (method) must be a string')
        assertion(calc.model['method'] in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq', 'fci'], \
                        'valid expansion methods (method) are: ccsd, ccsd(t), ccsdt, ccsdtq and fci')
        assertion(calc.model['cc_backend'] in ['pyscf', 'ecc', 'ncc'], \
                        'valid cc backends (cc_backend) are: pyscf, ecc and ncc')
        assertion(calc.model['solver'] in ['pyscf_spin0', 'pyscf_spin1'], \
                        'valid FCI solvers (solver) are: pyscf_spin0 and pyscf_spin1')
        assertion(isinstance(calc.model['hf_guess'], bool), \
                        'HF initial guess for FCI calcs (hf_guess) must be a bool')
        if calc.model['method'] != 'fci':
            assertion(calc.model['solver'] == 'pyscf_spin0', \
                            'setting a FCI solver for a non-FCI expansion model is not meaningful')
            assertion(calc.model['hf_guess'], \
                            'non-HF initial guess (hf_guess) only valid for FCI calcs')
            if calc.model['method'] == 'ccsdt':
                assertion(calc.model['cc_backend'] != 'pyscf', \
                            'ccsdt is not available with pyscf set as cc_backend')
            if calc.model['method'] == 'ccsdtq':
                assertion(calc.model['cc_backend'] == 'ncc', \
                            'ccsdtq is not available with pyscf or ecc set as cc_backend')
        if spin > 0:
            assertion(calc.model['solver'] != 'pyscf_spin0', \
                            'the pyscf_spin0 FCI solver is designed for spin singlets only')
            assertion(calc.model['cc_backend'] == 'pyscf', \
                            'the mbecc interface is designed for closed-shell systems only')
        # hf reference
        assertion(isinstance(calc.hf_ref['newton'], bool), \
                        'newton input in hf_ref dict (newton) must be a bool')
        assertion(isinstance(calc.hf_ref['symmetry'], (str, bool)), \
                        'HF symmetry input in hf_ref dict (symmetry) must be a str or bool')
        assertion(isinstance(calc.hf_ref['init_guess'], str), \
                        'HF initial guess in hf_ref dict (init_guess) must be a str')
        assertion(calc.hf_ref['init_guess'] in ['minao', 'atom', '1e'], \
                        'valid HF initial guesses in hf_ref dict (init_guess) are: minao, atom, and 1e')
        assertion(isinstance(calc.hf_ref['irrep_nelec'], dict), \
                        'occupation input in hf_ref dict (irrep_nelec) must be a dict')
        # reference model
        assertion(calc.ref['method'] in ['casci', 'casscf'], \
                        'valid reference models are: casci and casscf')
        assertion(calc.ref['active'] in ['manual', 'avas', 'pios'], \
                        'active space choices are currently: manuali, avas, or pios')
        assertion(isinstance(calc.ref['select'], list), \
                        'select key (select) for active space must be a list of orbitals')
        assertion(isinstance(calc.ref['ao-labels'], list), \
                        'list of ao labels (ao-labels) for avas space must be a list of ao strings')
        assertion(isinstance(calc.ref['pi-atoms'], list), \
                        'list of pi-space atoms (pi-atoms) for pios space must be a list of atomic indices (index-1 based)')
        assertion(isinstance(calc.ref['hf_guess'], bool), \
                        'HF initial guess for CASSCF calc (hf_guess) must be a bool')
        assertion(len(calc.ref['wfnsym']) == len(calc.ref['weights']), \
                        'list of wfnsym and weights for CASSCF calc (wfnsym/weights) must be of same length')
        assertion(isinstance(calc.ref['weights'], (tuple, list)), \
                        'weights for CASSCF calc (weights) must be a list of floats')
        assertion(all(isinstance(i, float) for i in calc.ref['weights']), \
                        'weights for CASSCF calc (weights) must be floats')
        assertion(abs(sum(calc.ref['weights']) - 1.) < 1.e-3, \
                        'sum of weights for CASSCF calc (weights) must be equal to 1.')
        if atom:
            if calc.ref['hf_guess']:
                assertion(len(set(calc.ref['wfnsym'])) == 1, \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
                assertion(calc.ref['wfnsym'][0] == symm.addons.irrep_id2name(symmetry, 0), \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
            for i in range(len(calc.ref['wfnsym'])):
                try:
                    calc.ref['wfnsym'][i] = symm.addons.irrep_name2id(symmetry, calc.ref['wfnsym'][i])
                except Exception as err:
                    raise ValueError('illegal choice of ref wfnsym -- PySCF error: {:}'.format(err))
        if calc.ref['active'] in ['avas', 'pios']:
           assertion(spin == 0, 'illegal active space selection algortihm for non-singlet system')
        # base model
        if calc.base['method'] is not None:
            assertion(calc.base['method'] in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'], \
                            'valid base models are currently: ccsd, ccsd(t), ccsdt and ccsdtq')
            if calc.base['method'] == 'ccsdt':
                assertion(calc.model['cc_backend'] != 'pyscf', \
                            'ccsdt is not available with pyscf set as cc_backend')
            if calc.base['method'] == 'ccsdtq':
                assertion(calc.model['cc_backend'] == 'ncc', \
                            'ccsdtq is not available with pyscf or ecc set as cc_backend')
        # state
        if atom:
            try:
                calc.state['wfnsym'] = symm.addons.irrep_name2id(symmetry, calc.state['wfnsym'])
            except Exception as err:
                raise ValueError('illegal choice of state wfnsym -- PySCF error: {:}'.format(err))
            assertion(calc.state['root'] >= 0, \
                            'choice of target state (root) must be an int >= 0')
            if calc.model['method'] != 'fci':
                assertion(calc.state['wfnsym'] == 0, \
                                'illegal choice of wfnsym for chosen expansion model')
                assertion(calc.state['root'] == 0, \
                                'excited states not implemented for chosen expansion model')
        # targets
        assertion(any(calc.target.values()) and len([x for x in calc.target.keys() if calc.target[x]]) == 1, \
                        'one and only one target property must be requested')
        assertion(all(isinstance(i, bool) for i in calc.target.values()), \
                        'values in target input (target) must be bools')
        assertion(set(list(calc.target.keys())) <= set(['energy', 'excitation', 'dipole', 'trans']), \
                        'invalid choice for target property. valid choices are: '
                        'energy, excitation energy (excitation), dipole, and transition dipole (trans)')
        if calc.target['excitation']:
            assertion(calc.state['root'] > 0, \
                            'calculation of excitation energy (excitation) requires target state root >= 1')
        if calc.target['trans']:
            assertion(calc.state['root'] > 0, \
                            'calculation of transition dipole moment (trans) requires target state root >= 1')
        if calc.model['cc_backend'] == 'ecc':
            assertion(calc.target['energy'], \
                            'calculation of targets other than energy are not possible using the ecc backend')
        # extra
        assertion(isinstance(calc.extra['pi_prune'], bool), \
                        'pruning of pi-orbitals (pi_prune) must be a bool')
        if calc.extra['pi_prune']:
            assertion(symm.addons.std_symb(symmetry) in ['D2h', 'C2v'], \
                            'pruning of pi-orbitals (pi_prune) is only implemented for linear D2h and C2v symmetries')
        # expansion thresholds
        assertion(isinstance(calc.thres['start'], int), \
                        'screening start order (start) must be an int')
        assertion(2 <= calc.thres['start'], \
                        'screening start order (start) must an int >= 2')
        assertion(isinstance(calc.thres['perc'], float), \
                        'screening thresholds (perc) must be a float')
        assertion(calc.thres['perc'] <= 1., \
                        'screening threshold (perc) must a float <= 1.')
        # orbital representation
        assertion(calc.orbs['type'] in ['can', 'local', 'ccsd', 'ccsd(t)'], \
                        'valid occupied orbital representations (occ) are currently: '
                        'canonical (can), pipek-mezey (local), or natural orbs (ccsd or ccsd(t))')
        if calc.orbs['type'] != 'can':
            assertion(calc.ref['method'] == 'casci', \
                            'non-canonical orbitals requires casci expansion reference')
        if atom and calc.orbs['type'] == 'local':
            assertion(symmetry == 'C1', \
                            'the combination of local orbs and point group symmetry '
                            'different from c1 is not allowed')
        # misc
        assertion(isinstance(calc.misc['order'], (int, type(None))), \
                        'maximum expansion order (order) must be an int >= 1')
        if calc.misc['order'] is not None:
            assertion(calc.misc['order'] >= 1, \
                            'maximum expansion order (order) must be an int >= 1')
            if len(calc.ref['select']) == 0:
                assertion(calc.misc['order'] >= 2, \
                                'maximum expansion order (order) must be an int >= 2 '
                                'for vacuum reference spaces')
        assertion(isinstance(calc.misc['rst'], bool), \
                        'restart logical (rst) must be a bool')
        assertion(isinstance(calc.misc['rst_freq'], int), \
                        'restart freqeuncy (rst_freq) must be an int')
        assertion(calc.misc['rst_freq'] >= 1, \
                        'restart frequency (rst_freq) must be an int >= 1')


