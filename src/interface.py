#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
interface module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import sys
import ctypes
import numpy as np
from pyscf import ao2mo, gto, scf, symm
from typing import Tuple

from tools import idx_tril, nelec

try:
    import settings
    cclib = ctypes.cdll.LoadLibrary(settings.MBECCLIB)
    CCLIB_AVAILABLE = True
except ImportError:
    CCLIB_AVAILABLE = False

MAX_MEM = 131071906
CONV_TOL = 10

def mbecc_interface(method: str, cc_backend: str, orb_type: str, point_group: str, \
                    orbsym: np.ndarray, h1e: np.ndarray, h2e: np.ndarray, \
                    n_elec: Tuple[int, int], higher_amp_extrap: bool, \
                    debug: int) -> Tuple[float, int]:
        """
        this function returns the results of a cc calculation using the mbecc
        interface

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> hf = scf.RHF(mol)
        >>> _ = hf.kernel()
        >>> cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        >>> hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        >>> h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
        >>> eri_ao = mol.intor('int2e_sph', aosym=4)
        >>> h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
        >>> h1e_cas = h1e[cas_idx[:, None], cas_idx]
        >>> cas_idx_tril = idx_tril(cas_idx)
        >>> h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        >>> n_elec = nelec(hf.mo_occ, cas_idx)
        >>> cc_energy, success = mbecc_interface('ccsd', 'ecc', 'can', 'C2v', orbsym[cas_idx], h1e_cas, \
                                                 h2e_cas, n_elec, False, 0)
        >>> np.isclose(cc_energy, -0.014118607610972705)
        True
        """
        # check for path to MBECC library
        if not CCLIB_AVAILABLE:
            msg = 'settings.py not found for module interface. ' + \
            f'Please create {os.path.join(os.path.dirname(__file__), "settings.py"):}\n'
            raise ModuleNotFoundError(msg)

        # method keys in cfour
        method_dict = {'ccsd': 10, 'ccsd(t)': 22, 'ccsdt': 18, 'ccsdtq': 46}

        # cc module
        cc_module_dict = {'ecc': 0, 'ncc': 1}

        # point group
        point_group_dict = {'C1': 1, 'C2': 2, 'Ci': 3, 'Cs': 4, 'D2': 5, 'C2v': 6, 'C2h': 7, 'D2h': 8}

        # settings
        method_val = ctypes.c_int64(method_dict[method])
        cc_module_val = ctypes.c_int64(cc_module_dict[cc_backend])
        point_group_val = ctypes.c_int64(point_group_dict[point_group])
        non_canonical = ctypes.c_int64(0 if orb_type == 'can' else 1)
        maxcor = ctypes.c_int64(MAX_MEM) # max memory in integer words
        conv = ctypes.c_int64(CONV_TOL)
        max_cycle = ctypes.c_int64(500)
        t3_extrapol = ctypes.c_int64(1 if higher_amp_extrap else 0)
        t4_extrapol = ctypes.c_int64(1 if higher_amp_extrap else 0)
        verbose = ctypes.c_int64(1 if debug >= 3 else 0)

        n_act = orbsym.size
        h2e = ao2mo.restore(1, h2e, n_act)

        # initialize variables
        n_elec_arr = np.array(n_elec, dtype=np.int64) # number of occupied orbitals
        n_act = ctypes.c_int64(n_act) # number of orbitals
        cc_energy = ctypes.c_double() # cc-energy output
        success = ctypes.c_int64() # success flag

        # perform cc calculation
        cclib.cc_interface(ctypes.byref(method_val), ctypes.byref(cc_module_val), \
                           ctypes.byref(non_canonical), ctypes.byref(maxcor), \
                           n_elec_arr.ctypes.data_as(ctypes.c_void_p), \
                           ctypes.byref(n_act), orbsym.ctypes.data_as(ctypes.c_void_p), \
                           ctypes.byref(point_group_val), \
                           h1e.ctypes.data_as(ctypes.c_void_p), \
                           h2e.ctypes.data_as(ctypes.c_void_p), ctypes.byref(conv), \
                           ctypes.byref(max_cycle), ctypes.byref(t3_extrapol), \
                           ctypes.byref(t4_extrapol), ctypes.byref(verbose), \
                           ctypes.byref(cc_energy), ctypes.byref(success))

        # convergence check
        if success.value != 1:

            # redo calculation in debug mode if not converged
            verbose = ctypes.c_int64(1)

            cclib.cc_interface(ctypes.byref(method_val), ctypes.byref(cc_module_val), \
                               ctypes.byref(non_canonical), ctypes.byref(maxcor), \
                               n_elec_arr.ctypes.data_as(ctypes.c_void_p), \
                               ctypes.byref(n_act), orbsym.ctypes.data_as(ctypes.c_void_p), \
                               ctypes.byref(point_group_val), \
                               h1e.ctypes.data_as(ctypes.c_void_p), \
                               h2e.ctypes.data_as(ctypes.c_void_p), ctypes.byref(conv), \
                               ctypes.byref(max_cycle), ctypes.byref(t3_extrapol), \
                               ctypes.byref(t4_extrapol), ctypes.byref(verbose), \
                               ctypes.byref(cc_energy), ctypes.byref(success))

        return cc_energy.value, success.value

if __name__ == "__main__":
    import doctest
    doctest.testmod()


