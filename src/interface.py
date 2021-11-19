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

try:
    import settings
    import imp
    PyCheMPS2 = imp.load_dynamic('PyCheMPS2', settings.PYCHEMPS2BIN)
    PYCHEMPS2_AVAILABLE = True
except ImportError:
    PYCHEMPS2_AVAILABLE = False

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

        return cc_energy.value, success.value


# point group ID defined in CheMPS2, see
# http://sebwouters.github.io/CheMPS2/classCheMPS2_1_1Irreps.html
GROUPNAME_ID = { 
    'C1' : 0,
    'Ci' : 1,
    'C2' : 2,
    'Cs' : 3,
    'D2' : 4,
    'C2v': 5,
    'C2h': 6,
    'D2h': 7,
}


def chemps2_interface(mol, calc, h1e, eri):
        """
        this function returns the results of a DMRG calculation with PyCheMPS2
        """
        # check for path to PyCheMPS2 library
        if not PYCHEMPS2_AVAILABLE:
            msg = 'settings.py not found for module interface. ' + \
            f'Please create {os.path.join(os.path.dirname(__file__), "settings.py"):}\n'
            raise ModuleNotFoundError(msg)

        ncas = mol.norb - mol.ncore
        nelec = mol.nelectron - 2 * mol.ncore

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # get core and cas indices
        core_idx = np.arange(mol.ncore)
        cas_idx = np.arange(mol.ncore, mol.norb)

        # get h2e indices
        cas_idx_tril = idx_tril(cas_idx)

        # get h2e_cas
        h2e = eri[cas_idx_tril[:, None], cas_idx_tril]

        # init core energy
        e_core = mol.e_nuc

        # determine effective core fock potential
        if core_idx.size > 0:
            core_vhf = np.sum(vhf[core_idx], axis=0)
        else:
            core_vhf = 0

        # calculate core energy
        e_core += np.trace((hcore + .5 * core_vhf)[core_idx[:, None], core_idx]) * 2.

        # extract cas integrals
        h1e = (hcore + core_vhf)[cas_idx[:, None], cas_idx]
        
        if mol.symmetry:
            groupNumber = GROUPNAME_ID[mol.groupname]
        else:
            groupNumber = 0

        orbsym = calc.orbsym[mol.ncore:]

        wfn_irrep = 0
        dmrg_states = [ 200 , 500 , 1000 , 1000 ]
        dmrg_noise = [ 1 , 1 , 1 , 0 ]
        dmrg_e_convergence = 1e-8
        dmrg_noise_factor = 0.03
        dmrg_maxiter_noise = 5
        dmrg_maxiter_silent = 100

        Initializer = PyCheMPS2.PyInitialize()
        Initializer.Init()

        Ham = PyCheMPS2.PyHamiltonian(ncas, groupNumber,
                                        np.asarray(orbsym, dtype=np.int32))
        eri = ao2mo.restore(1, h2e, ncas)
        for i in range(ncas):
            for j in range(ncas):
                totsym = orbsym[i] ^ orbsym[j]
                if 0 == totsym:
                    Ham.setTmat(i, j, h1e[i,j])
                for k in range(ncas):
                    for l in range(ncas):
                        totsym = orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l]
                        if 0 == totsym:
                            Ham.setVmat(i, k, j, l, eri[i,j,k,l])
        Ham.setEconst(0)

        if isinstance(nelec, (int, np.integer)):
            spin2 = 0
        else:
            spin2 = (nelec[0]-nelec[1])
            nelec = sum(nelec)

        Prob = PyCheMPS2.PyProblem(Ham, spin2, nelec, wfn_irrep)
        Prob.SetupReorderD2h()

        OptScheme = PyCheMPS2.PyConvergenceScheme(len(dmrg_states))
        for cnt, m in enumerate(dmrg_states):
            if dmrg_noise[cnt]:
                OptScheme.setInstruction(cnt, m, dmrg_e_convergence,
                                         dmrg_maxiter_noise,
                                         dmrg_noise_factor)
            else:
                OptScheme.setInstruction(cnt, m, dmrg_e_convergence,
                                         dmrg_maxiter_silent, 0.0)

        theDMRG = PyCheMPS2.PyDMRG(Prob, OptScheme)

        Energy = theDMRG.Solve() + e_core
        theDMRG.calc2DMandCorrelations()

        mut_info = np.empty((ncas,)*2)
        for i in range(ncas):
            for j in range(ncas):
                mut_info[i,j] = theDMRG.getMutInfo(i, j)

        theDMRG.deleteStoredOperators()

        # The order of deallocation matters!
        del(theDMRG)
        del(OptScheme)
        del(Prob)
        del(Ham)
        del(Initializer)

        return Energy, mut_info

if __name__ == "__main__":
    import doctest
    doctest.testmod()


