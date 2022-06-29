#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
wrapper module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
import scipy as sc
from pyscf import gto, scf, symm, lo, cc, mcscf, fci, ao2mo
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_rdm
from pyscf.lib.exceptions import PointGroupSymmetryError
from copy import copy
from typing import TYPE_CHECKING, cast, Union
from warnings import catch_warnings, simplefilter

from pymbe.kernel import main_kernel, dipole_kernel, cc_kernel, e_core_h1e
from pymbe.tools import (
    RDMCls,
    assertion,
    mat_idx,
    near_nbrs,
    core_cas,
    get_vhf,
    get_nelec,
    get_nhole,
    get_nexc,
    idx_tril,
    ground_state_sym,
    get_occup,
    get_symm_op_matrices,
    transform_mos,
)

if TYPE_CHECKING:

    from typing import Tuple, Dict, List, Optional, Any


HF_CONV_TOL = 1.0e-10
CAS_CONV_TOL = 1.0e-10
LOC_CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05
COORD_TOL = 1.0e-05
MO_SYMM_TOL = 1.0e-13
DETECT_MO_SYMM_TOL = 1.0e-01


def ints(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    x2c: bool = False,
    u: float = 1.0,
    matrix: Tuple[int, int] = (1, 6),
    pbc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function returns 1e and 2e mo integrals and effective fock potentials from
    individual occupied orbitals
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "ints: mol (first argument) must be a gto.Mole object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "ints: mo coefficients (second argument) must be a np.ndarray",
    )
    # x2c
    assertion(
        isinstance(x2c, bool),
        "ints: spin-free x2c relativistic hamiltonian option (x2c keyword argument) "
        "must be a bool",
    )
    # hubbard
    if not mol.atom:
        # matrix
        assertion(
            isinstance(matrix, tuple)
            and len(matrix) == 2
            and isinstance(matrix[0], int)
            and isinstance(matrix[1], int),
            "ints: hubbard matrix (matrix keyword argument) must be a tuple of ints "
            "with a dimension of 2",
        )
        # u parameter
        assertion(
            isinstance(u, float),
            "ints: hubbard on-site repulsion parameter (u keyword argument) must be a "
            "float",
        )
        assertion(
            u > 0.0,
            "ints: only repulsive hubbard models are implemented, hubbard on-site "
            "repulsion parameter (u keyword argument) must be > 0.",
        )
        # periodic boundary conditions
        assertion(
            isinstance(pbc, bool),
            "ints: hubbard model periodic boundary conditions (pbc keyword argument) "
            "must be a bool",
        )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # hcore_ao and eri_ao w/o symmetry
    hcore_ao, eri_ao = _ao_ints(mol, x2c, u, matrix, pbc)

    # compute hcore
    hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

    # eri_mo w/o symmetry
    eri = ao2mo.incore.full(eri_ao, mo_coeff)

    # compute vhf
    vhf = get_vhf(eri, nocc, norb)

    # restore 4-fold symmetry in eri
    eri = ao2mo.restore(4, eri, norb)

    return hcore, eri, vhf


def _ao_ints(
    mol: gto.Mole, x2c: bool, u: float, matrix: Tuple[int, int], pbc: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns 1e and 2e ao integrals
    """
    if mol.atom:

        # hcore_ao
        if x2c:
            hf = scf.ROHF(mol).x2c()
        else:
            hf = scf.ROHF(mol)
        hcore = hf.get_hcore()
        # eri_ao w/o symmetry
        if mol.cart:
            eri = mol.intor("int2e_cart", aosym=1)
        else:
            eri = mol.intor("int2e_sph", aosym=1)

    else:

        # hcore_ao
        hcore = _hubbard_h1e(matrix, pbc)
        # eri_ao
        eri = _hubbard_eri(matrix, u)

    return hcore, eri


def _hubbard_h1e(matrix: Tuple[int, int], pbc: bool = False) -> np.ndarray:
    """
    this function returns the hubbard hopping hamiltonian
    """
    # dimension
    if 1 in matrix:
        ndim = 1
    else:
        ndim = 2

    # nsites
    nsites = matrix[0] * matrix[1]

    # init h1e
    h1e = np.zeros([nsites] * 2, dtype=np.float64)

    if ndim == 1:

        # adjacent neighbours
        for i in range(nsites - 1):
            h1e[i, i + 1] = h1e[i + 1, i] = -1.0

        if pbc:
            h1e[-1, 0] = h1e[0, -1] = -1.0

    elif ndim == 2:

        # number of x- and y-sites
        nx, ny = matrix[0], matrix[1]

        # adjacent neighbours
        for site_1 in range(nsites):

            site_1_xy = mat_idx(site_1, nx, ny)
            nbrs = near_nbrs(site_1_xy, nx, ny)

            for site_2 in range(site_1):

                site_2_xy = mat_idx(site_2, nx, ny)

                if site_2_xy in nbrs:
                    h1e[site_1, site_2] = h1e[site_2, site_1] = -1.0

    return h1e


def _hubbard_eri(matrix: Tuple[int, int], u: float) -> np.ndarray:
    """
    this function returns the hubbard two-electron hamiltonian
    """
    # nsites
    nsites = matrix[0] * matrix[1]

    # init eri
    eri = np.zeros([nsites] * 4, dtype=np.float64)

    # compute eri
    for i in range(nsites):
        eri[i, i, i, i] = u

    return eri


def dipole_ints(
    mol: gto.Mole, mo_coeff: np.ndarray, gauge_origin: np.ndarray
) -> np.ndarray:
    """
    this function returns dipole integrals (in AO basis)
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "dipole_ints: mol (first argument) must be a gto.Mole object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "dipole_ints: mo coefficients (second argument) must be a np.ndarray",
    )
    # gauge origin
    assertion(
        isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3,
        "dipole_ints: gauge origin (gauge_origin keyword argument) must be a "
        "np.ndarray of size 3",
    )

    with mol.with_common_origin(gauge_origin):
        dipole = mol.intor_symmetric("int1e_r", comp=3)

    return np.einsum("pi,xpq,qj->xij", mo_coeff, dipole, mo_coeff)


def hf(
    mol: gto.Mole,
    target: str = "energy",
    init_guess: str = "minao",
    newton: bool = False,
    irrep_nelec: Dict[str, Any] = {},
    x2c: bool = False,
    u: float = 1.0,
    matrix: Tuple[int, int] = (1, 6),
    pbc: bool = False,
    gauge_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Tuple[
    scf.hf.SCF,
    Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    """
    this function returns the results of a restricted (open-shell) hartree-fock
    calculation
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole), "hf: mol (first argument) must be a gto.Mole object"
    )
    # init_guess
    assertion(
        isinstance(init_guess, str),
        "hf: hf initial guess (init_guess keyword argument) must be a str",
    )
    assertion(
        init_guess in ["minao", "atom", "1e"],
        "hf: valid hf initial guesses (init_guess keyword argument) are: "
        "minao, atom, and 1e",
    )
    # newton
    assertion(
        isinstance(newton, bool),
        "hf: newton option (newton keyword argument) must be a bool",
    )
    # irrep_nelec
    assertion(
        isinstance(irrep_nelec, dict),
        "hf: irreducible representation occupation (irrep_nelec keyword argument) must "
        "be a dict",
    )
    # x2c
    assertion(
        isinstance(x2c, bool),
        "hf: spin-free x2c relativistic hamiltonian option (x2c keyword argument) must "
        "be a bool",
    )
    # hubbard
    if not mol.atom:
        # matrix
        assertion(
            isinstance(matrix, tuple)
            and len(matrix) == 2
            and isinstance(matrix[0], int)
            and isinstance(matrix[1], int),
            "hf: hubbard matrix (matrix keyword argument) must be a tuple of ints with "
            "a dimension of 2",
        )
        # u parameter
        assertion(
            isinstance(u, float),
            "hf: hubbard on-site repulsion parameter (u keyword argument) must be a "
            "float",
        )
        assertion(
            u > 0.0,
            "hf: only repulsive hubbard models are implemented, hubbard on-site "
            "repulsion parameter (u keyword argument) must be > 0.",
        )

        # periodic boundary conditions
        assertion(
            isinstance(pbc, bool),
            "hf: hubbard model periodic boundary conditions (pbc keyword argument) "
            "must be a bool",
        )
    # gauge origin
    assertion(
        isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3,
        "hf: gauge origin (gauge_origin keyword argument) must be a np.ndarray of size "
        "3",
    )

    # initialize restricted hf calc
    if x2c:
        hf = scf.RHF(mol).x2c()
    else:
        hf = scf.RHF(mol)

    hf.init_guess = init_guess
    if newton:
        hf.conv_tol = 1.0e-01
    else:
        hf.conv_tol = HF_CONV_TOL
    hf.max_cycle = 1000

    if mol.atom:
        # ab initio hamiltonian
        hf.irrep_nelec = irrep_nelec
    else:
        # model hamiltonian
        hf.get_ovlp = lambda *args: np.eye(matrix[0] * matrix[1])
        hf.get_hcore = lambda *args: _hubbard_h1e(matrix, pbc)
        hf._eri = _hubbard_eri(matrix, u)

    # hf calc
    with catch_warnings():
        simplefilter("ignore")
        hf.kernel()

    if newton:

        # initial mo coefficients and occupation
        mo_coeff = hf.mo_coeff
        mo_occ = hf.mo_occ

        # new so-hf object
        hf = hf.newton()
        hf.conv_tol = HF_CONV_TOL

        with catch_warnings():
            simplefilter("ignore")
            hf.kernel(mo_coeff, mo_occ)

    # store occupation and orbsym
    occup = hf.mo_occ
    norb = occup.size
    if mol.symmetry:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
    else:
        orbsym = np.zeros(norb, dtype=np.int64)

    hf_prop: Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]

    if target == "energy":

        hf_prop = hf.e_tot.item()

    elif target == "excitation":

        hf_prop = 0.0

    elif target == "dipole":

        if mol.atom:
            dm = hf.make_rdm1()
            if mol.spin > 0:
                dm = dm[0] + dm[1]
            with mol.with_common_orig(gauge_origin):
                ao_dip = mol.intor_symmetric("int1e_r", comp=3)
            hf_prop = np.einsum("xij,ji->x", ao_dip, dm)
        else:
            hf_prop = np.zeros(3, dtype=np.float64)

    elif target == "trans":

        hf_prop = np.zeros(3, dtype=np.float64)

    elif target == "rdm12":

        rdm1 = np.zeros(2 * (norb,), dtype=np.float64)
        np.einsum("ii->i", rdm1)[...] += occup

        rdm2 = np.zeros(4 * (norb,), dtype=np.float64)
        occup_a = occup.copy()
        occup_a[occup_a > 0.0] = 1.0
        occup_b = occup - occup_a
        # d_ppqq = k_pa*k_qa + k_pb*k_qb + k_pa*k_qb + k_pb*k_qa = k_p*k_q
        np.einsum("iijj->ij", rdm2)[...] += np.einsum("i,j", occup, occup)
        # d_pqqp = - (k_pa*k_qa + k_pb*k_qb)
        np.einsum("ijji->ij", rdm2)[...] += np.einsum(
            "i,j", occup_a, occup_a
        ) + np.einsum("i,j", occup_b, occup_b)

        hf_prop = (rdm1, rdm2)

    return hf, hf_prop, orbsym, np.asarray(hf.mo_coeff, order="C")


def ref_mo(
    orbs: str,
    mol: gto.Mole,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    orbsym: np.ndarray,
    ncore: int,
    ref_space: np.ndarray = np.array([]),
    wfnsym: Optional[List[Union[str, int]]] = None,
    weights: List[float] = [1.0],
    hf_guess: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns a set of reference mo coefficients and symmetries plus the
    associated spaces
    """
    # orbs
    assertion(
        isinstance(orbs, str),
        "ref_mo: orbital representation (first argument) must be a str",
    )
    assertion(
        orbs in ["ccsd", "ccsd(t)", "local", "casscf"],
        "ref_mo: valid orbital representations (first argument) are: natural (ccsd or "
        "ccsd(t)), pipek-mezey (local), or casscf orbs (casscf)",
    )
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "ref_mo: mol (second argument) must be a gto.Mole object",
    )
    # hf
    assertion(
        isinstance(hf, scf.hf.SCF),
        "ref_mo: hf (third argument) must be a scf.hf.SCF object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "ref_mo: mo coefficients (fourth argument) must be a np.ndarray",
    )
    # orbsym_can
    assertion(
        isinstance(orbsym, np.ndarray),
        "ref_mo: orbital symmetry (fifth argument) must be a np.ndarray",
    )
    # ncore
    assertion(
        isinstance(ncore, int),
        "ref_mo: number of core orbitals (sixth argument) must be an int",
    )
    # casscf
    if orbs == "casscf":
        # set default casscf reference symmetry
        if wfnsym is None:
            wfnsym = (
                [symm.addons.irrep_id2name(mol.groupname, 0)] if mol.groupname else [0]
            )
        # ref_space
        assertion(
            isinstance(ref_space, np.ndarray),
            "ref_mo: reference space (ref_space keyword argument) must be a np.ndarray "
            "of orbital indices",
        )
        assertion(
            np.any(np.isin(np.arange(max(mol.nelec)), ref_space)),
            "ref_mo: no singly/doubly occupied orbitals in cas space (ref_space "
            "keyword argument) of casscf calculation",
        )
        assertion(
            np.any(np.isin(np.arange(min(mol.nelec), mol.nao), ref_space)),
            "ref_mo: no singly occupied/virtual orbitals in cas space (ref_space "
            "keyword argument) of casscf calculation",
        )
        # wfnsym
        assertion(
            isinstance(wfnsym, list) and all(isinstance(i, str) for i in wfnsym),
            "ref_mo: casscf wavefunction symmetries (wfnsym keyword argument) must be "
            "a list of str",
        )
        wfnsym_int: List[int] = []
        for i in range(len(wfnsym)):
            try:
                wfnsym_int.append(
                    symm.addons.irrep_name2id(mol.groupname, wfnsym[i])
                    if mol.groupname
                    else 0
                )
            except Exception as err:
                raise ValueError(
                    "ref_mo: illegal choice of ref wfnsym (wfnsym keyword argument) "
                    f"-- PySCF error: {err}"
                )
        # weights
        assertion(
            isinstance(weights, list) and all(isinstance(i, float) for i in weights),
            "ref_mo: casscf weights (weights keyword argument) must be a list of "
            "floats",
        )
        assertion(
            len(wfnsym_int) == len(weights),
            "ref_mo: list of wfnsym (wfnsym keyword argument) and weights (weights "
            "keyword argument) for casscf calculation must be of same length",
        )
        assertion(
            all(isinstance(i, float) for i in weights),
            "ref_mo: casscf weights (weights keyword argument) must be floats",
        )
        assertion(
            abs(sum(weights) - 1.0) < 1.0e-3,
            "ref_mo: sum of weights for casscf calculation (weights keyword argument) "
            "must be equal to 1.",
        )
        # hf_guess
        assertion(
            isinstance(hf_guess, bool),
            "ref_mo: hf initial guess (hf_guess keyword argument) must be a bool",
        )
        if hf_guess:
            assertion(
                len(set(wfnsym_int)) == 1,
                "ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) "
                "when enforcing hf initial guess (hf_guess keyword argument) because "
                "wfnsym should be limited to one state",
            )
            assertion(
                wfnsym_int[0] == ground_state_sym(orbsym, mol.nelec, mol.groupname),
                "ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) "
                "when enforcing hf initial guess (hf_guess keyword argument) because "
                "wfnsym does not equal hf state symmetry",
            )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # occup
    occup = get_occup(norb, mol.nelec)

    # copy mo coefficients
    mo_coeff_out = np.copy(mo_coeff)

    # set core and cas spaces
    core_idx, cas_idx = core_cas(nocc, np.arange(ncore, nocc), np.arange(nocc, norb))

    # NOs
    if orbs in ["ccsd", "ccsd(t)"]:

        # compute rmd1
        ccsd = cc.CCSD(hf)
        frozen_orbs = np.asarray(
            [i for i in range(hf.mo_coeff.shape[1]) if i not in cas_idx]
        )
        if frozen_orbs.size > 0:
            ccsd.frozen = frozen_orbs

        # settings
        ccsd.conv_tol = 1.0e-10
        ccsd.conv_tol_normt = ccsd.conv_tol
        ccsd.max_cycle = 500
        ccsd.async_io = False
        ccsd.diis_start_cycle = 4
        ccsd.diis_space = 12
        ccsd.incore_complete = True
        eris = ccsd.ao2mo()

        # calculate ccsd energy
        ccsd.kernel(eris=eris)

        # convergence check
        assertion(
            ccsd.converged,
            f"CCSD error: no convergence, core_idx = {core_idx}, cas_idx = {cas_idx}",
        )

        # rdm1
        if orbs == "ccsd":
            ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
            rdm1 = ccsd.make_rdm1()
        elif orbs == "ccsd(t)":
            if mol.spin == 0:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

        if mol.spin > 0:
            rdm1 = rdm1[0] + rdm1[1]

        # occupied - occupied block
        mask = occup == 2.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # singly occupied - singly occupied block
        mask = occup == 1.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # virtual - virtual block
        mask = occup == 0.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # orbital symmetries
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

    # pipek-mezey localized orbitals
    elif orbs == "local":

        # occupied - occupied block
        mask = occup == 2.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = LOC_CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # singly occupied - singly occupied block
        mask = occup == 1.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = LOC_CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # virtual - virtual block
        mask = occup == 0.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = LOC_CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # orbital symmetries
        if mol.symmetry:
            orbsym = np.zeros(norb, dtype=np.int64)

    # casscf
    elif orbs == "casscf":

        # electrons in active space
        act_nelec = get_nelec(occup, ref_space)

        # sorter for active space
        n_core_inact = np.array(
            [i for i in range(nocc) if i not in ref_space], dtype=np.int64
        )
        n_virt_inact = np.array(
            [a for a in range(nocc, norb) if a not in ref_space], dtype=np.int64
        )
        sort_casscf = np.concatenate((n_core_inact, ref_space, n_virt_inact))
        mo_coeff_casscf = mo_coeff_out[:, sort_casscf]

        # update orbsym
        if mol.symmetry:
            orbsym_casscf = symm.label_orb_symm(
                mol, mol.irrep_id, mol.symm_orb, mo_coeff_casscf
            )

        # run casscf
        mo_coeff_out = _casscf(
            mol,
            wfnsym_int,
            weights,
            orbsym_casscf,
            hf_guess,
            hf,
            mo_coeff_casscf,
            ref_space,
            act_nelec,
            ncore,
        )

        # reorder mo_coeff
        mo_coeff_out = mo_coeff_out[:, np.argsort(sort_casscf)]

        # orbital symmetries
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

    return np.asarray(mo_coeff_out, order="C"), orbsym


class _hubbard_PM(lo.pipek.PM):
    """
    this class constructs the site-population tensor for each orbital-pair density
    see: pyscf example - 40-hubbard_model_PM_localization.py
    """

    def atomic_pops(
        self, mol: gto.Mole, mo_coeff: np.ndarray, method: Optional[str] = None
    ) -> np.ndarray:
        """
        this function overwrites the tensor used in the pm cost function and its gradients
        """
        return np.einsum("pi,pj->pij", mo_coeff, mo_coeff)


def _casscf(
    mol: gto.Mole,
    wfnsym: List[int],
    weights: List[float],
    orbsym: np.ndarray,
    hf_guess: bool,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    ref_space: np.ndarray,
    nelec: np.ndarray,
    ncore: int,
) -> np.ndarray:
    """
    this function returns the results of a casscf calculation
    """
    # init casscf
    cas = mcscf.CASSCF(hf, ref_space.size, nelec)

    # casscf settings
    cas.conv_tol = CAS_CONV_TOL
    cas.max_cycle_macro = 500
    cas.frozen = ncore

    # init fcisolver
    if nelec[0] == nelec[1]:
        fcisolver = fci.direct_spin0_symm.FCI(mol)
    else:
        fcisolver = fci.direct_spin1_symm.FCI(mol)

    # create unique list of wfnsym while maintaining order
    unique_wfnsym = list(dict.fromkeys(wfnsym))

    # fci settings
    fcisolver.conv_tol = CAS_CONV_TOL
    fcisolver.orbsym = orbsym[ref_space]
    fcisolver.wfnsym = unique_wfnsym[0]
    cas.fcisolver = fcisolver

    # state-averaged casscf
    if len(wfnsym) > 1:

        if len(unique_wfnsym) == 1:

            # state average over all states of same symmetry
            cas.state_average_(weights)

        else:

            # nroots for first fcisolver
            fcisolver.nroots = np.count_nonzero(np.asarray(wfnsym) == unique_wfnsym[0])

            # init list of fcisolvers
            fcisolvers = [fcisolver]

            # loop over symmetries
            for i in range(1, len(unique_wfnsym)):

                # copy fcisolver
                fcisolver_ = copy(fcisolver)

                # wfnsym for fcisolver_
                fcisolver_.wfnsym = unique_wfnsym[i]

                # nroots for fcisolver_
                fcisolver_.nroots = np.count_nonzero(
                    np.asarray(wfnsym) == unique_wfnsym[i]
                )

                # append to fcisolvers
                fcisolvers.append(fcisolver_)

            # state average
            mcscf.state_average_mix_(cas, fcisolvers, weights)

    # hf starting guess
    if hf_guess:
        na = fci.cistring.num_strings(ref_space.size, nelec[0])
        nb = fci.cistring.num_strings(ref_space.size, nelec[1])
        ci0 = np.zeros((na, nb))
        ci0[0, 0] = 1
    else:
        ci0 = None

    # run casscf calc
    cas.kernel(mo_coeff, ci0=ci0)

    # collect ci vectors
    if len(wfnsym) == 1:
        c = [cas.ci]
    else:
        c = cas.ci

    # multiplicity check
    for root in range(len(c)):

        s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)

        if abs((mol.spin + 1) - mult) > SPIN_TOL:

            # fix spin by applyting level shift
            sz = abs(nelec[0] - nelec[1]) * 0.5
            cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.0))

            # run casscf calc
            cas.kernel(mo_coeff, ci0=ci0)

            # collect ci vectors
            if len(wfnsym) == 1:
                c = [cas.ci]
            else:
                c = cas.ci

            # verify correct spin
            for root in range(len(c)):
                s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
                assertion(
                    abs((mol.spin + 1) - mult) < SPIN_TOL,
                    f"spin contamination for root entry = {root}, 2*S + 1 = {mult:.6f}",
                )

    # convergence check
    assertion(cas.converged, "CASSCF error: no convergence")

    return np.asarray(cas.mo_coeff, order="C")


def ref_prop(
    mol: gto.Mole,
    hcore: np.ndarray,
    eri: np.ndarray,
    ref_space: np.ndarray,
    method: str = "fci",
    base_method: Optional[str] = None,
    cc_backend: str = "pyscf",
    target: str = "energy",
    orbsym: Optional[np.ndarray] = None,
    fci_state_sym: Optional[Union[str, int]] = None,
    fci_state_root: int = 0,
    hf_guess: bool = True,
    hf_prop: Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    vhf: Optional[np.ndarray] = None,
    dipole_ints: Optional[np.ndarray] = None,
    orb_type: str = "can",
):
    """
    this function returns reference space properties
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "ref_prop: mol (first argument) must be a gto.Mole object",
    )
    # hcore
    assertion(
        isinstance(hcore, np.ndarray),
        "ref_prop: core hamiltonian integrals (second argument) must be a np.ndarray",
    )
    # eri
    assertion(
        isinstance(eri, np.ndarray),
        "ref_prop: electron repulsion integrals (third argument) must be a np.ndarray",
    )
    # ref_space
    assertion(
        isinstance(ref_space, np.ndarray),
        "ref_prop: reference space (fourth argument) must be a np.ndarray of orbital "
        "indices",
    )
    # method
    assertion(
        isinstance(method, str),
        "ref_prop: electronic structure method (method keyword argument) must be a str",
    )
    assertion(
        method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq", "fci"],
        "ref_prop: valid electronic structure methods (method keyword argument) are: "
        "ccsd, ccsd(t), ccsdt, ccsdtq and fci",
    )
    # base_method
    assertion(
        isinstance(base_method, (str, type(None))),
        "ref_prop: base model electronic structure method (base_method keyword "
        "argument) must be a str or None",
    )
    if base_method is not None:
        assertion(
            base_method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"],
            "ref_prop: valid base model electronic structure methods (base_method "
            "keyword argument) are: ccsd, ccsd(t), ccsdt and ccsdtq",
        )
    # orbsym
    if orbsym is None:
        orbsym = np.zeros(mol.nao.item(), dtype=np.int64)
    assertion(
        isinstance(orbsym, np.ndarray),
        "ref_prop: orbital symmetry (orbsym keyword argument) must be a np.ndarray",
    )
    # fci
    if fci_state_sym is None:
        fci_state_sym = ground_state_sym(orbsym, mol.nelec, mol.groupname)
    if method == "fci":
        # fci_state_sym
        assertion(
            isinstance(fci_state_sym, (str, int)),
            "ref_prop: fci state symmetry (fci_state_sym keyword argument) must be a "
            "str or int",
        )
        if isinstance(fci_state_sym, str):
            try:
                fci_state_sym = (
                    symm.addons.irrep_name2id(mol.groupname, fci_state_sym)
                    if mol.groupname
                    else 0
                )
            except Exception as err:
                raise ValueError(
                    "ref_prop: illegal choice of fci state symmetry (fci_state_sym "
                    f"keyword argument) -- PySCF error: {err}"
                )
        # fci_state_root
        assertion(
            isinstance(fci_state_root, int),
            "ref_prop: fci state root (fci_state_root keyword argument) must be an int",
        )
        assertion(
            fci_state_root >= 0,
            "ref_prop: choice of fci target state (fci_state_root keyword argument) "
            "must be an int >= 0",
        )
        # hf_guess
        assertion(
            isinstance(hf_guess, bool),
            "ref_prop: hf initial guess (hf_guess keyword argument) must be a bool",
        )
        if hf_guess:
            assertion(
                fci_state_sym == ground_state_sym(orbsym, mol.nelec, mol.groupname),
                "ref_prop: illegal choice of reference wfnsym (wfnsym keyword "
                "argument) when enforcing hf initial guess (hf_guess keyword argument) "
                "because wfnsym does not equal hf state symmetry",
            )
    # cc methods
    elif method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"] or base_method:
        assertion(
            isinstance(cc_backend, str),
            "ref_prop: coupled-cluster backend (cc_backend keyword argument) must be a "
            "str",
        )
        assertion(
            cc_backend in ["pyscf", "ecc", "ncc"],
            "ref_prop: valid coupled-cluster backends (cc_backend keyword argument) "
            "are: pyscf, ecc and ncc",
        )
        if base_method == "ccsdt":
            assertion(
                cc_backend != "pyscf",
                "ref_prop: ccsdt is not available with pyscf coupled-cluster backend "
                "(cc_backend keyword argument)",
            )
        if base_method == "ccsdtq":
            assertion(
                cc_backend == "ncc",
                "ref_prop: ccsdtq is not available with pyscf and ecc coupled-cluster "
                "backends (cc_backend keyword argument)",
            )
        if mol.spin > 0:
            assertion(
                cc_backend == "pyscf",
                "ref_prop: open-shell systems are not available with ecc and ncc "
                "coupled-cluster backends (cc_backend keyword argument)",
            )
    # target
    assertion(
        isinstance(target, str),
        "ref_prop: target property (target keyword argument) must be str",
    )
    assertion(
        target in ["energy", "excitation", "dipole", "trans", "rdm12"],
        "ref_prop: valid target properties (target keyword argument) are: energy, "
        "excitation energy (excitation), dipole, and transition dipole (trans)",
    )
    if target in ["excitation", "trans"]:
        assertion(
            fci_state_root > 0,
            "ref_prop: calculation of excitation energies or transition dipole moments "
            "(target keyword argument) requires target state root (state_root keyword "
            "argument) >= 1",
        )
    if method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"] or base_method:
        assertion(
            target in ["energy", "dipole", "rdm12"],
            "ref_prop: calculation of excitation energies or transition dipole moments "
            "(target keyword argument) not possible with coupled-cluster methods "
            "(method keyword argument)",
        )
        if cc_backend in ["ecc", "ncc"]:
            assertion(
                target == "energy",
                "ref_prop: calculation of targets (target keyword argument) other than "
                "energy not possible with ecc and ncc coupled-cluster backends "
                "(cc_backend keyword argument)",
            )
    # hf_prop
    if target == "energy":
        assertion(
            isinstance(hf_prop, float),
            "ref_prop: hartree-fock energy (hf_prop keyword argument) must be a float",
        )
    elif target == "dipole":
        assertion(
            isinstance(hf_prop, np.ndarray),
            "ref_prop: hartree-fock dipole moment (hf_prop keyword argument) must be a "
            "np.ndarray",
        )
    elif target == "rdm12":
        assertion(
            isinstance(hf_prop, tuple)
            and len(hf_prop) == 2
            and isinstance(hf_prop[0], np.ndarray)
            and isinstance(hf_prop[1], np.ndarray),
            "ref_prop: hartree-fock 1- and 2-particle density matrices (hf_prop "
            "keyword argument) must be a tuple of np.ndarray with dimension 2",
        )
    # vhf
    if vhf is not None:
        assertion(
            isinstance(vhf, np.ndarray),
            "ref_prop: hartree-fock potential (vhf keyword argument) must be a "
            "np.ndarray",
        )
    # dipole_ints
    if target in ["dipole", "trans"]:
        assertion(
            isinstance(dipole_ints, np.ndarray),
            "ref_prop: dipole integrals (dipole_ints keyword argument) must be a "
            "np.ndarray",
        )
    # orbital representation
    assertion(
        isinstance(orb_type, str),
        "ref_prop: orbital representation (orbs keyword argument) must be a str",
    )
    assertion(
        orb_type in ["can", "ccsd", "ccsd(t)", "local", "casscf"],
        "ref_prop: valid orbital representations (orbs keyword argument) are: "
        "canonical (can), natural (ccsd or ccsd(t)), pipek-mezey (local), or casscf "
        "orbs (casscf)",
    )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # occup
    occup = get_occup(norb, mol.nelec)

    # hf_prop
    if target == "excitation":
        hf_prop = 0.0
    elif target == "trans":
        hf_prop = np.zeros(3, dtype=np.float64)

    # core_idx and cas_idx
    core_idx, cas_idx = core_cas(nocc, ref_space, np.array([], dtype=np.int64))

    # compute vhf
    if vhf is None:
        vhf = get_vhf(eri, nocc, norb)

    # nelec
    nelec = get_nelec(occup, cas_idx)

    # nhole
    nhole = get_nhole(nelec, cas_idx)

    # nexc
    nexc = get_nexc(nelec, nhole)

    # ref_prop
    ref_prop: Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]

    if (
        nexc <= 1
        or (base_method in ["ccsd", "ccsd(t)"] and nexc <= 2)
        or (base_method == "ccsdt" and nexc <= 3)
        or (base_method == "ccsdtq" and nexc <= 4)
    ):

        # no correlation in expansion reference space
        if target in ["energy", "excitation"]:
            ref_prop = 0.0
        elif target in ["dipole", "trans"]:
            ref_prop = np.zeros(3, dtype=np.float64)
        elif target == "rdm12":
            ref_prop = (
                np.zeros(2 * (ref_space.size,), dtype=np.float64),
                np.zeros(4 * (ref_space.size,), dtype=np.float64),
            )

    else:

        # get cas_space h2e
        cas_idx_tril = idx_tril(cas_idx)
        h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

        # compute e_core and h1e_cas
        e_core, h1e_cas = e_core_h1e(
            mol.energy_nuc().item(), hcore, vhf, core_idx, cas_idx
        )

        # exp model
        res = main_kernel(
            method,
            cc_backend,
            orb_type,
            mol.spin,
            occup,
            target,
            cast(int, fci_state_sym),
            mol.groupname,
            orbsym,
            hf_guess,
            fci_state_root,
            RDMCls(hf_prop[0], hf_prop[1])
            if isinstance(hf_prop, tuple)
            else cast(Union[float, np.ndarray], hf_prop),
            e_core,
            h1e_cas,
            h2e_cas,
            core_idx,
            cas_idx,
            nelec,
            0,
            higher_amp_extrap=False,
        )

        if target in ["energy", "excitation"]:

            ref_prop = res[target]

        elif target == "dipole":

            ref_prop = dipole_kernel(
                cast(np.ndarray, dipole_ints),
                occup,
                cas_idx,
                res["rdm1"],
                hf_dipole=cast(np.ndarray, hf_prop),
            )

        elif target == "trans":

            ref_prop = dipole_kernel(
                cast(np.ndarray, dipole_ints),
                occup,
                cas_idx,
                res["t_rdm1"],
                trans=True,
            )

        elif target == "rdm12":

            ref_prop = (res["rdm1"], res["rdm2"])

        # base model
        if base_method is not None:

            res = cc_kernel(
                mol.spin,
                occup,
                core_idx,
                cas_idx,
                base_method,
                cc_backend,
                nelec,
                orb_type,
                mol.groupname,
                orbsym,
                h1e_cas,
                h2e_cas,
                False,
                target,
                0,
            )

            if target == "energy":

                ref_prop -= res[target]

            elif target == "dipole":

                ref_prop -= dipole_kernel(
                    cast(np.ndarray, dipole_ints),
                    occup,
                    cas_idx,
                    res["rdm1"],
                    hf_dipole=cast(np.ndarray, hf_prop),
                )

            elif target == "rdm12":

                ref_prop = (
                    cast(tuple, ref_prop)[0] - res["rdm1"],
                    cast(tuple, ref_prop)[1] - res["rdm2"],
                )

    return ref_prop


def base(
    method: str,
    mol: gto.Mole,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    ncore: int,
    cc_backend: str = "pyscf",
    target: str = "energy",
    orbsym: Optional[np.ndarray] = None,
    hf_prop: Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    gauge_origin: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    this function returns base model energy
    """
    # method
    assertion(
        isinstance(method, str),
        "base: base model electronic structure method (first argument) must be a str",
    )
    assertion(
        method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"],
        "base: valid base model electronic structure methods (first argument) are: "
        "ccsd, ccsd(t), ccsdt, and ccsdtq",
    )
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "base: mol (second argument) must be a gto.Mole object",
    )
    # hf
    assertion(
        isinstance(hf, scf.hf.SCF),
        "base: hf (third argument) must be a scf.hf.SCF object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "base: mo coefficients (fourth argument) must be a np.ndarray",
    )
    # ncore
    assertion(
        isinstance(ncore, int),
        "base: number of core orbitals (fifth argument) must be an int",
    )
    # cc_backend
    assertion(
        isinstance(cc_backend, str),
        "base: coupled-cluster backend (cc_backend keyword argument) must be a str",
    )
    assertion(
        cc_backend in ["pyscf", "ecc", "ncc"],
        "base: valid coupled-cluster backends (cc_backend keyword argument) are: "
        "pyscf, ecc and ncc",
    )
    if method == "ccsdt":
        assertion(
            cc_backend != "pyscf",
            "base: ccsdt (first argument) is not available with pyscf coupled-cluster "
            "backend (cc_backend keyword argument)",
        )
    if method == "ccsdtq":
        assertion(
            cc_backend == "ncc",
            "base: ccsdtq (first argument) is not available with pyscf and ecc "
            "coupled-cluster backends (cc_backend keyword argument)",
        )
    if mol.spin > 0:
        assertion(
            cc_backend == "pyscf",
            "base: open-shell systems are not available with ecc and ncc "
            "coupled-cluster backends (cc_backend keyword argument)",
        )
    # target
    assertion(
        isinstance(target, str),
        "base: target property (target keyword argument) must be str",
    )
    assertion(
        target in ["energy", "dipole", "rdm12"],
        "base: valid target properties (keyword argument) with coupled-cluster base "
        "methods are: energy (energy) and dipole moment (dipole)",
    )
    if cc_backend in ["ecc", "ncc"]:
        assertion(
            target == "energy",
            "base: calculation of targets (target keyword argument) other than energy "
            "are not possible with ecc and ncc coupled-cluster backends (cc_backend "
            "keyword argument)",
        )
    # orbsym
    if orbsym is None:
        orbsym = np.zeros(mol.nao.item(), dtype=np.int64)
    assertion(
        isinstance(orbsym, np.ndarray),
        "base: orbital symmetry (orbsym keyword argument) must be a np.ndarray",
    )
    if target == "dipole":
        # hf_dipole
        assertion(
            isinstance(hf_prop, np.ndarray),
            "base: hartree-fock dipole moment (hf_prop keyword argument) must be a "
            "np.ndarray",
        )
        # gauge_dipole
        assertion(
            isinstance(gauge_origin, np.ndarray),
            "base: gauge origin (gauge_origin keyword argument) must be a np.ndarray",
        )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # occup
    occup = get_occup(norb, mol.nelec)

    # hcore_ao and eri_ao with 8-fold symmetry
    hcore_ao = hf.get_hcore()
    eri_ao = hf._eri

    # remove symmetry from eri_ao
    eri_ao = ao2mo.restore(1, eri_ao, norb)

    # compute hcore
    hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

    # eri_mo w/o symmetry
    eri = ao2mo.incore.full(eri_ao, mo_coeff)

    # compute vhf for core orbitals
    vhf = get_vhf(eri, ncore, norb)

    # restore 4-fold symmetry in eri_mo
    eri = ao2mo.restore(4, eri, norb)

    # compute dipole integrals
    if target == "dipole" and mol.atom:
        dip_ints = dipole_ints(mol, mo_coeff, cast(np.ndarray, gauge_origin))
    else:
        dip_ints = None

    # set core and correlated spaces
    core_idx, corr_idx = core_cas(nocc, np.arange(ncore, nocc), np.arange(nocc, norb))

    # get correlated space h2e
    corr_idx_tril = idx_tril(corr_idx)
    h2e_corr = eri[corr_idx_tril[:, None], corr_idx_tril]

    # determine effective core fock potential
    if core_idx.size > 0:
        core_vhf = np.sum(vhf, axis=0)
    else:
        core_vhf = 0.0

    # get effective h1e for correlated space
    h1e_corr = (hcore + core_vhf)[corr_idx[:, None], corr_idx]

    # nelec
    nelec = get_nelec(occup, corr_idx)

    # run calc
    res = cc_kernel(
        mol.spin,
        occup,
        core_idx,
        corr_idx,
        method,
        cc_backend,
        nelec,
        "can",
        mol.groupname,
        orbsym,
        h1e_corr,
        h2e_corr,
        False,
        target,
        0,
    )

    # collect results
    if target == "energy":
        base_prop = res["energy"]
    elif target == "dipole":
        base_prop = dipole_kernel(
            cast(np.ndarray, dip_ints),
            occup,
            corr_idx,
            res["rdm1"],
            hf_dipole=cast(np.ndarray, hf_prop),
        )
    elif target == "rdm12":
        base_prop = res["rdm1"], res["rdm2"]

    return base_prop


def linear_orbsym(mol: gto.Mole, mo_coeff: np.ndarray) -> np.ndarray:
    """
    returns orbsym in linear point groups for pi pruning
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "linear orbsym: mol (first argument) must be a gto.Mole object",
    )
    assertion(
        symm.addons.std_symb(mol.groupname) in ["D2h", "C2v"],
        "linear orbsym: only works for linear D2h and C2v symmetries",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "linear orbsym: mo coefficients (second argument) must be a np.ndarray",
    )

    # recast mol in parent point group (dooh/coov)
    mol_parent = mol.copy()
    parent_group = "Dooh" if mol.groupname == "D2h" else "Coov"
    mol_parent = mol_parent.build(0, 0, symmetry=parent_group)

    orbsym_parent = symm.label_orb_symm(
        mol_parent, mol_parent.irrep_id, mol_parent.symm_orb, mo_coeff
    )

    return orbsym_parent


def symm_eqv_mo(
    mol: gto.Mole, mo_coeff: np.ndarray, point_group: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns an array of permutations of symmetry equivalent orbitals for each
    symmetry operation
    """
    # convert point group to standard symbol
    point_group = symm.std_symb(point_group)

    # get different equivalent atom types
    atom_types = [
        np.array(atom_type)
        for atom_type in gto.mole.atom_types(mol._atom, mol.basis).values()
    ]

    # get atom coords
    coords = mol.atom_coords()

    # get symmetry origin and axes
    symm_orig, symm_axes = _get_symm_coord(point_group, mol._atom, mol._basis)

    # shift coordinates to symmetry origin
    coords -= symm_orig

    # rotate coordinates to symmetry axes
    coords = (symm_axes.T @ coords.T).T

    # get number of occupied orbitals
    nocc = max(mol.nelec)

    # get ao shell offsets
    ao_loc = mol.ao_loc_nr()

    # get angular momentum for each shell
    l_shell = [mol.bas_angular(shell) for shell in range(mol.nbas)]

    # get Wigner D matrices to rotate mo coefficients from input coordinate system to
    # symmetry axes
    Ds = symm.basis._ao_rotation_matrices(mol, symm_axes)

    # get ao offset for every atom
    _, _, ao_start_list, ao_stop_list = mol.offset_nr_by_atom().T

    # get list of all symmetry operation matrices for point group
    ops = get_symm_op_matrices(point_group, max(l_shell))

    # get ao overlap matrix
    sao = gto.intor_cross("int1e_ovlp", mol, mol)

    # diagonalize ao overlap matrix
    e, v = sc.linalg.eigh(sao)

    # get all indices for non-zero eigenvalues
    idx = e > 1e-15

    # calculate root of ao overlap matrix
    sqrt_sao = (v[:, idx] * np.sqrt(e[idx])) @ v[:, idx].conj().T

    # calculate reciprocal root of ao overlap matrix
    rec_sqrt_sao = (v[:, idx] / np.sqrt(e[idx])) @ v[:, idx].conj().T

    # transform mo coefficients into orthogonal ao basis
    mo_coeff = sqrt_sao @ mo_coeff

    # initialize atom indices permutation array
    permut_atom_idx = np.empty(mol.natm, dtype=np.int64)

    # initialize ao indices permutation array for every symmetry operation
    permut_ao_idx = np.empty((len(ops), mol.nao), dtype=np.int64)

    # initialize array of equivalent mos
    symm_eqv_mos = np.empty((len(ops), mo_coeff.shape[0]), dtype=np.int64)

    # loop over symmetry operations
    for op, (cart_op_mat, sph_op_mats) in enumerate(ops):

        # loop over group of equivalent atoms with equivalent basis functions
        for atom_type in atom_types:

            # extract coordinates of atom type
            atom_coords = coords[atom_type]

            # get indices necessary to sort coords lexicographically
            lex_idx = symm.geom.argsort_coords(atom_coords)

            # sort coords lexicographically
            lex_coords = atom_coords[lex_idx]

            # get indices necessary to sort atom numbers
            sort_idx = np.argsort(lex_idx)

            # get new coordinates of atoms after applying symmetry operation
            new_atom_coords = (cart_op_mat.T @ atom_coords.T).T

            # get indices necessary to sort new coords lexicographically
            lex_idx = symm.geom.argsort_coords(new_atom_coords)

            # check whether rearranged new coords are the same as rearranged original
            # coords
            if not np.allclose(lex_coords, new_atom_coords[lex_idx], atol=COORD_TOL):
                raise PointGroupSymmetryError("Symmetry identical atoms not found")

            # reorder indices according to sort order of original indices
            op_atom_ids = lex_idx[sort_idx]

            # add atom permutations for atom type
            permut_atom_idx[atom_type] = atom_type[op_atom_ids]

        # loop over atoms
        for atom_id, permut_atom_id in enumerate(permut_atom_idx):

            # add ao permutations for atom
            permut_ao_idx[
                op, ao_start_list[atom_id] : ao_stop_list[atom_id]
            ] = np.arange(ao_start_list[permut_atom_id], ao_stop_list[permut_atom_id])

    # loop over repeating symmetrization and orthogonalization steps until convergence
    # an iterative procedure is needed because the symmetric (Löwdin) orthogonalization
    # between blocks of symmetry equivalent orbitals destroys some symmetry
    for _ in range(100):

        # intitialze symmetrized mo coefficients
        symm_mo_coeff = np.zeros_like(mo_coeff)

        # initialize maximum deviation from symmetry
        max_asymm = 0.0

        # define outer product of occupied orbitals
        project_occ = np.einsum("ij,kj->ik", mo_coeff[:, :nocc], mo_coeff[:, :nocc])

        # define outer product of virtual orbitals
        project_virt = np.einsum("ij,kj->ik", mo_coeff[:, nocc:], mo_coeff[:, nocc:])

        # loop over symmetry operations
        for op, (cart_op_mat, sph_op_mats) in enumerate(ops):

            # rotate symmetry operation matrices for spherical harmonics to original
            # coordinates and back
            rot_sph_op_mats = [
                rot_mat.T @ op_mat @ rot_mat for rot_mat, op_mat in zip(Ds, sph_op_mats)
            ]

            # permute aos
            op_mo_coeff = mo_coeff[permut_ao_idx[op], :]

            # transform mos
            op_mo_coeff = transform_mos(op_mo_coeff, rot_sph_op_mats, l_shell, ao_loc)

            # consider only occupied components of transformed occupied orbitals
            op_mo_coeff[:, :nocc] = np.einsum(
                "ij,jk->ik", project_occ, op_mo_coeff[:, :nocc]
            )

            # consider only virtual components of transformed virtual orbitals
            op_mo_coeff[:, nocc:] = np.einsum(
                "ij,jk->ik", project_virt, op_mo_coeff[:, nocc:]
            )

            # calculate overlaps between transformed mo coefficients and actual mo
            # coefficients
            overlaps = np.einsum("ij,ik->jk", op_mo_coeff.conj(), mo_coeff)

            # get index of maximum overlap for every mo
            symm_eqv_mos[op, :] = np.argmax(np.abs(overlaps), axis=1)

            # get maximum overlaps
            eqv_overlaps = np.abs(np.diag(overlaps[:, symm_eqv_mos[op, :]]))

            # get boolean array for equivalent mos above threshold
            eqv_idx = np.isclose(eqv_overlaps, 1.0, atol=DETECT_MO_SYMM_TOL)

            # only consider equivalent mos above threshold
            symm_eqv_mos[op, np.logical_not(eqv_idx)] = -1

            # get maximum deviation from symmetry for all symmetry equivalent orbitals
            max_asymm = max(np.max(1.0 - eqv_overlaps[eqv_idx]), max_asymm)

            # average symmetry equivalent orbitals
            symm_mo_coeff[:, symm_eqv_mos[op, eqv_idx]] += (
                np.sign(overlaps[eqv_idx, symm_eqv_mos[op, eqv_idx]])
                * op_mo_coeff[:, eqv_idx]
            )

        # check if maximum deviation from symmetry is below threshold
        if max_asymm < MO_SYMM_TOL:

            # symmetrization finished
            break

        # normalize mo coefficients
        symm_mo_coeff /= np.linalg.norm(symm_mo_coeff, axis=0)

        # orthogonalize mo coefficients
        symm_mo_coeff = lo.orth.vec_lowdin(symm_mo_coeff)

        # set orthogonalized orbitals as new orbitals
        mo_coeff = symm_mo_coeff

    # symmetrization did not converge
    else:

        raise PointGroupSymmetryError(
            "Symmetrization of localized orbitals did not converge."
        )

    # backtransform mo coefficients
    mo_coeff = rec_sqrt_sao @ mo_coeff

    return symm_eqv_mos, mo_coeff


def _get_symm_coord(
    point_group: str,
    atoms: List[List[Union[str, float]]],
    basis: Dict[str, List[List[Union[int, List[float]]]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function determines the charge center and symmetry axes for a given point group
    """
    # initialize symmetry object
    rawsys = symm.SymmSys(atoms, basis)

    # determine charge center of molecule
    charge_center = rawsys.charge_center

    # initialize boolean for correct point group
    correct_symm = False

    # 3D rotation group
    if point_group == "SO3":

        symm_axes = np.eye(3)

    # proper cyclic groups Cn
    elif point_group[0] == "C" and point_group[1:].isnumeric():

        if point_group[1:] == 1:

            correct_symm = True
            symm_axes = np.eye(3)

        else:

            tot_main_rot = int(point_group[1:])

            zaxis, n = rawsys.search_c_highest()

            for axis in np.eye(3):
                if not symm.parallel_vectors(axis, zaxis):
                    symm_axes = symm.geom._make_axes(zaxis, axis)
                    break

            if n % tot_main_rot == 0:
                correct_symm = True
                symm_axes = symm.geom._refine(symm_axes)

    # improper cyclic group Ci
    elif point_group == "Ci":

        if rawsys.has_icenter():
            correct_symm = True
            symm_axes = np.eye(3)

    # improper cyclic group Cs
    elif point_group == "Cs":

        mirror = rawsys.search_mirrorx(None, 1)

        if mirror is not None:
            correct_symm = True
            symm_axes = symm.geom._make_axes(mirror, np.array((1.0, 0.0, 0.0)))

    # improper cyclic group Sn
    elif point_group[0] == "S":

        tot_main_rot = int(point_group[1:])

        zaxis, n = rawsys.search_c_highest()

        for axis in np.eye(3):
            if not symm.parallel_vectors(axis, zaxis):
                symm_axes = symm.geom._make_axes(zaxis, axis)
                break

        if (2 * n) % tot_main_rot == 0 and rawsys.has_improper_rotation(
            symm_axes[2], n
        ):
            correct_symm = True
            symm_axes = symm.geom._refine(symm_axes)

    # dihedral groups Dn
    elif point_group[0] == "D" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        zaxis, n = rawsys.search_c_highest()

        c2x = rawsys.search_c2x(zaxis, n)

        if n % tot_main_rot == 0 and c2x is not None:
            correct_symm = True
            symm_axes = symm.geom._refine(symm.geom._make_axes(zaxis, c2x))

    # Dnh
    elif point_group[0] == "D" and point_group[-1] == "h":

        if point_group[1:-1] == "oo":

            w1, u1 = rawsys.cartesian_tensor(1)

            if (
                np.allclose(w1[:2], 0, atol=symm.TOLERANCE / np.sqrt(1 + len(atoms)))
                and rawsys.has_icenter()
            ):
                correct_symm = True
                symm_axes = u1.T

        else:

            tot_main_rot = int(point_group[1:-1])

            zaxis, n = rawsys.search_c_highest()

            c2x = rawsys.search_c2x(zaxis, n)

            if n % tot_main_rot == 0 and c2x is not None:
                symm_axes = symm.geom._make_axes(zaxis, c2x)
                if rawsys.has_mirror(symm_axes[2]):
                    correct_symm = True
                    symm_axes = symm.geom._refine(symm_axes)

    # Dnd
    elif point_group[0] == "D" and point_group[-1] == "d":

        tot_main_rot = int(point_group[1:-1])

        zaxis, n = rawsys.search_c_highest()

        c2x = rawsys.search_c2x(zaxis, n)

        if n % tot_main_rot == 0 and c2x is not None:
            symm_axes = symm.geom._make_axes(zaxis, c2x)
            if rawsys.has_improper_rotation(symm_axes[2], n):
                correct_symm = True
                symm_axes = symm.geom._refine(symm_axes)

    # Cnv
    elif point_group[0] == "C" and point_group[-1] == "v":

        if point_group[1:-1] == "oo":

            _, u1 = rawsys.cartesian_tensor(1)

            if np.allclose(w1[:2], 0, atol=symm.TOLERANCE / np.sqrt(1 + len(atoms))):
                correct_symm = True
                symm_axes = u1.T

        else:

            tot_main_rot = int(point_group[1:-1])

        zaxis, n = rawsys.search_c_highest()

        mirrorx = rawsys.search_mirrorx(zaxis, n)

        if n % tot_main_rot == 0 and mirrorx is not None:
            correct_symm = True
            symm_axes = symm.geom._refine(symm.geom._make_axes(zaxis, mirrorx))

    # Cnh
    elif point_group[0] == "C" and point_group[-1] == "h":

        tot_main_rot = int(point_group[1:-1])

        zaxis, n = rawsys.search_c_highest()

        for axis in np.eye(3):
            if not symm.parallel_vectors(axis, zaxis):
                symm_axes = symm.geom._make_axes(zaxis, axis)
                break

        if n % tot_main_rot == 0 and rawsys.has_mirror(symm_axes[2]):
            correct_symm = True
            symm_axes = symm.geom._refine(symm_axes)

    # cubic groups
    elif point_group in ["T", "Th", "Td", "O", "Oh"]:

        group_name, symm_axes = symm._search_ot_group(rawsys)

        if group_name is not None:
            correct_symm = True
            symm_axes = symm.geom._refine(symm_axes)

    # icosahedral groups
    elif point_group in ["I", "Ih"]:

        group_name, symm_axes = symm._search_i_group(rawsys)

        if group_name is not None:
            correct_symm = True
            symm_axes = symm.geom._refine(symm_axes)

    # check if molecule has symmetry of point group
    if correct_symm:

        return charge_center, symm_axes

    else:

        raise PointGroupSymmetryError(
            "Molecule does not have supplied symmetry. Maybe try "
            "reducing symmetry tolerance."
        )
