#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy testing module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import pytest
import numpy as np
from math import comb
from itertools import chain, product, combinations
from mpi4py import MPI
from typing import TYPE_CHECKING

from pymbe.energy import EnergyExpCls

if TYPE_CHECKING:
    from pyscf import gto, scf
    from typing import List, Tuple, Optional

    from pymbe.pymbe import MBE

test_cases_mbe = [
    (
        "h2o",
        1,
        -0.0374123708341898,
        -0.0018267714680604286,
        -0.0374123708341898,
        -0.004676546354273725,
        0.0018267714680604286,
        0.010886635891736773,
    ),
    (
        "h2o",
        2,
        -0.11605435599270209,
        -0.0001698239845069338,
        -0.15346672682557028,
        -0.004144798428310789,
        0.0001698239845069338,
        0.009556269221292268,
    ),
]

test_cases_ref_prop = [
    ("h2o", "fci", None, "pyscf", 0, -0.03769780809258805, 1.0, 0.9856854425080487),
    ("h2o", "ccsd", None, "pyscf", 0, -0.03733551374348559, 1.0, 1.0),
    (
        "h2o",
        "fci",
        "ccsd",
        "pyscf",
        0,
        -0.00036229313775759664,
        1.0,
        0.9856854425080487,
    ),
    ("h2o", "ccsd(t)", "ccsd", "pyscf", 0, -0.0003336954549769955, 1.0, 1.0),
    ("h2o", "ccsd", None, "ecc", 0, -0.03733551374348559, 1.0, 1.0),
    ("h2o", "fci", "ccsd", "ecc", 0, -0.0003622938195746786, 1.0, 0.9856854425080487),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "ecc",
        0,
        -0.0003336954549769955,
        1.0,
        1.0,
    ),
    ("h2o", "ccsd", None, "ncc", 0, -0.03733551374348559, 1.0, 1.0),
    ("h2o", "fci", "ccsd", "ncc", 0, -0.0003622938195746786, 1.0, 0.9856854425080487),
    ("h2o", "ccsd(t)", "ccsd", "ncc", 0, -0.0003336954549769955, 1.0, 1.0),
]

test_cases_kernel = [
    ("h2o", "fci", "pyscf", -0.00627368491326763),
    ("hubbard", "fci", "pyscf", -2.8759428090050676),
    ("h2o", "ccsd", "pyscf", -0.006273684840715439),
    ("h2o", "ccsd", "ecc", -0.00627368488758955),
    ("h2o", "ccsd", "ncc", -0.006273684885561386),
]

test_cases_fci_kernel = [
    ("h2o", -0.00627368491326763, 1.0, 0.9979706719796727),
    ("hubbard", -2.875942809005066, 1.0, 0.14031179591440068),
]

test_cases_cc_kernel = [
    ("h2o", "ccsd", "pyscf", -0.0062736848407002966),
    ("h2o", "ccsd(t)", "pyscf", -0.0062736848407002966),
    ("h2o", "ccsd", "ecc", -0.00627368488758955),
    ("h2o", "ccsd(t)", "ecc", -0.006273684887573003),
    ("h2o", "ccsdt", "ecc", -0.00627368488758955),
    ("h2o", "ccsd", "ncc", -0.006273684885561386),
    ("h2o", "ccsd(t)", "ncc", -0.006273684885577932),
    ("h2o", "ccsdt", "ncc", -0.006273684885561386),
    ("h2o", "ccsdtq", "ncc", -0.006273684885577932),
]


@pytest.fixture
def exp(mol: gto.Mole, hf: scf.RHF, mbe: MBE):
    """
    this fixture constructs a EnergyExpCls object
    """
    exp = EnergyExpCls(mbe)
    exp.target = "energy"

    return exp


@pytest.mark.parametrize(
    argnames="system, order, ref_inc_sum, ref_inc_amax, ref_mbe_tot_prop, "
    "ref_mean_inc, ref_min_inc, ref_max_inc",
    argvalues=test_cases_mbe,
    ids=["-".join([case[0], str(case[1])]) for case in test_cases_mbe],
    indirect=["system"],
)
def test_mbe(
    mbe: MBE,
    exp: EnergyExpCls,
    hf: scf.RHF,
    nocc: int,
    order: int,
    ref_inc_sum: float,
    ref_inc_amax: float,
    ref_mbe_tot_prop: np.ndarray,
    ref_mean_inc: float,
    ref_min_inc: float,
    ref_max_inc: float,
) -> None:
    """
    this function tests _mbe
    """
    hashes: List[List[np.ndarray]] = []
    inc: List[List[np.ndarray]] = []

    for exp.order in range(1, order + 1):
        hashes.append([])
        inc.append([])

        exp.n_incs.append(np.empty(exp.order + 1, dtype=np.int64))
        for k in range(exp.order + 1):
            exp.n_incs[-1][k] = comb(
                exp.exp_space[-1][exp.exp_space[-1] < nocc].size, k
            ) * comb(exp.exp_space[-1][nocc <= exp.exp_space[-1]].size, exp.order - k)

        exp.n_tuples["van"].append(np.sum(exp.n_incs[-1]))

        exp._mbe(mbe.mpi)

        for k in range(exp.order + 1):
            if exp.hashes[-1][k] is not None:
                hashes[-1].append(
                    np.ndarray(
                        buffer=exp.hashes[-1][k].Shared_query(0)[0],  # type: ignore
                        dtype=np.int64,
                        shape=(exp.n_incs[exp.order - 1][k],),
                    )
                )
            else:
                hashes[-1].append(
                    np.empty(shape=(exp.n_incs[exp.order - 1][k],), dtype=np.int64)
                )

            if exp.incs[-1][k] is not None:
                inc[-1].append(
                    np.ndarray(
                        buffer=exp.incs[-1][k].Shared_query(0)[0],  # type: ignore
                        dtype=np.float64,
                        shape=(exp.n_incs[exp.order - 1][k], 1),
                    )
                )
            else:
                inc[-1].append(
                    np.empty(shape=(exp.n_incs[exp.order - 1][k], 1), dtype=np.float64)
                )

            exp.hashes.append(exp.hashes[-1])

            exp.incs.append(exp.incs[-1])

            exp.mean_inc.append(exp.mean_inc[-1])
            exp.min_inc.append(exp.min_inc[-1])
            exp.max_inc.append(exp.max_inc[-1])

    assert all(
        [isinstance(item, MPI.Win) for item in exp.hashes[-1] if item is not None]
    )
    assert all([isinstance(item, MPI.Win) for item in exp.incs[-1] if item is not None])
    for k in range(exp.order + 1):
        assert np.array_equal(
            hashes[-1][k],
            np.sort(
                [
                    hash(
                        np.fromiter(chain.from_iterable(tup), dtype=np.int64).tobytes()
                    )
                    for tup in product(
                        combinations(exp.exp_space[-1][exp.exp_space[-1] < nocc], k),
                        combinations(
                            exp.exp_space[-1][nocc <= exp.exp_space[-1]], exp.order - k
                        ),
                    )
                ]
            ),
        )
    assert np.sum(np.concatenate(inc[-1])) == pytest.approx(ref_inc_sum)
    assert np.amax(np.concatenate(inc[-1])) == pytest.approx(ref_inc_amax)
    assert exp.mbe_tot_prop[-1] == pytest.approx(ref_mbe_tot_prop)
    assert exp.mean_inc[-1] == pytest.approx(ref_mean_inc)
    assert exp.min_inc[-1] == pytest.approx(ref_min_inc)
    assert exp.max_inc[-1] == pytest.approx(ref_max_inc)


@pytest.mark.parametrize(
    argnames="system",
    argvalues=["h2o"],
    indirect=["system"],
)
def test_purge(mbe: MBE, exp: EnergyExpCls) -> None:
    """
    this function tests _purge
    """
    start_tups = [
        [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        [
            np.array([], dtype=np.int64),
            np.array(
                [
                    np.array([0, 3], dtype=np.int64),
                    np.array([0, 4], dtype=np.int64),
                    np.array([0, 5], dtype=np.int64),
                    np.array([1, 3], dtype=np.int64),
                    np.array([1, 4], dtype=np.int64),
                    np.array([1, 5], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                    np.array([2, 4], dtype=np.int64),
                    np.array([2, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array([], dtype=np.int64),
        ],
        [
            np.array([], dtype=np.int64),
            np.array(
                [
                    np.array([0, 3, 4], dtype=np.int64),
                    np.array([0, 3, 5], dtype=np.int64),
                    np.array([0, 4, 5], dtype=np.int64),
                    np.array([1, 3, 4], dtype=np.int64),
                    np.array([1, 3, 5], dtype=np.int64),
                    np.array([1, 4, 5], dtype=np.int64),
                    np.array([2, 3, 4], dtype=np.int64),
                    np.array([2, 3, 5], dtype=np.int64),
                    np.array([2, 4, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    np.array([0, 1, 3], dtype=np.int64),
                    np.array([0, 1, 4], dtype=np.int64),
                    np.array([0, 1, 5], dtype=np.int64),
                    np.array([0, 2, 3], dtype=np.int64),
                    np.array([0, 2, 4], dtype=np.int64),
                    np.array([0, 2, 5], dtype=np.int64),
                    np.array([1, 2, 3], dtype=np.int64),
                    np.array([1, 2, 4], dtype=np.int64),
                    np.array([1, 2, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array([], dtype=np.int64),
        ],
        [
            np.array([], dtype=np.int64),
            np.array(
                [
                    np.array([0, 3, 4, 5], dtype=np.int64),
                    np.array([1, 3, 4, 5], dtype=np.int64),
                    np.array([2, 3, 4, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    np.array([0, 1, 3, 4], dtype=np.int64),
                    np.array([0, 1, 3, 5], dtype=np.int64),
                    np.array([0, 1, 4, 5], dtype=np.int64),
                    np.array([0, 2, 3, 4], dtype=np.int64),
                    np.array([0, 2, 3, 5], dtype=np.int64),
                    np.array([0, 2, 4, 5], dtype=np.int64),
                    np.array([1, 2, 3, 4], dtype=np.int64),
                    np.array([1, 2, 3, 5], dtype=np.int64),
                    np.array([1, 2, 4, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    np.array([0, 1, 2, 3], dtype=np.int64),
                    np.array([0, 1, 2, 4], dtype=np.int64),
                    np.array([0, 1, 2, 5], dtype=np.int64),
                ],
                dtype=np.int64,
            ),
            np.array([], dtype=np.int64),
        ],
    ]

    exp.nocc = 3
    exp.occup = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    exp.exp_space = [np.array([0, 1, 2, 3, 5], dtype=np.int64)]
    exp.screen_orbs = np.array([4], dtype=np.int64)
    exp.order = 4
    exp.n_incs = [
        np.array([0, 0], dtype=np.int64),
        np.array([0, 9, 0], dtype=np.int64),
        np.array([0, 9, 9, 0], dtype=np.int64),
        np.array([0, 3, 9, 3, 0], dtype=np.int64),
    ]

    start_hashes = [
        [
            np.sort(
                np.array([hash(tup.tobytes()) for tup in nocc_tups], dtype=np.int64)
            )
            for nocc_tups in order_tups
        ]
        for order_tups in start_tups
    ]

    hashes: List[List[np.ndarray]] = []
    incs: List[List[np.ndarray]] = []

    for order in range(1, 5):
        k = order - 1
        hashes.append([])
        incs.append([])
        exp.hashes.append([])
        exp.incs.append([])
        for l in range(order + 1):
            hashes_win = MPI.Win.Allocate_shared(
                8 * exp.n_incs[k][l], 8, comm=mbe.mpi.local_comm
            )
            buf = hashes_win.Shared_query(0)[0]
            hashes[-1].append(
                np.ndarray(
                    buffer=buf,  # type: ignore
                    dtype=np.int64,
                    shape=(exp.n_incs[k][l],),
                )
            )
            hashes[-1][l][:] = start_hashes[k][l]
            exp.hashes[-1].append(hashes_win)

            inc_win = MPI.Win.Allocate_shared(
                8 * exp.n_incs[k][l], 8, comm=mbe.mpi.local_comm
            )
            buf = inc_win.Shared_query(0)[0]
            incs[-1].append(
                np.ndarray(
                    buffer=buf,  # type: ignore
                    dtype=np.float64,
                    shape=(exp.n_incs[k][l],),
                )
            )
            incs[-1][l][:] = np.arange(1, exp.n_incs[k][l] + 1, dtype=np.float64)
            exp.incs[-1].append(inc_win)

    exp._purge(mbe.mpi)

    purged_hashes: List[List[np.ndarray]] = []
    purged_incs: List[List[np.ndarray]] = []

    for order in range(1, 5):
        k = order - 1
        purged_hashes.append([])
        purged_incs.append([])
        for l in range(order + 1):
            purged_hashes_win = exp.hashes[k][l]
            if purged_hashes_win is not None:
                buf = purged_hashes_win.Shared_query(0)[0]
                purged_hashes[-1].append(
                    np.ndarray(
                        buffer=buf,  # type: ignore
                        dtype=np.int64,
                        shape=(exp.n_incs[k][l],),
                    )
                )
            else:
                purged_hashes[-1].append(
                    np.empty(shape=(exp.n_incs[k][l],), dtype=np.int64)
                )

            purged_inc_win = exp.incs[k][l]
            if purged_inc_win is not None:
                buf = purged_inc_win.Shared_query(0)[0]
                purged_incs[-1].append(
                    np.ndarray(
                        buffer=buf,  # type: ignore
                        dtype=np.float64,
                        shape=(exp.n_incs[k][l],),
                    )
                )
            else:
                purged_incs[-1].append(
                    np.empty(shape=(exp.n_incs[k][l],), dtype=np.float64)
                )

    assert np.array_equal(exp.n_incs[0], np.array([0, 0], dtype=np.int64))
    assert np.array_equal(exp.n_incs[1], np.array([0, 6, 0], dtype=np.int64))
    assert np.array_equal(exp.n_incs[2], np.array([0, 3, 6, 0], dtype=np.int64))
    assert np.array_equal(exp.n_incs[3], np.array([0, 0, 3, 2, 0], dtype=np.int64))
    for order_idx in range(exp.order):
        for nocc_idx in range(exp.n_incs[order_idx].size):
            sort_idx = np.argsort(
                np.array(
                    [hash(tup.tobytes()) for tup in start_tups[order_idx][nocc_idx]],
                    dtype=np.int64,
                )
            )
            assert np.array_equal(
                purged_hashes[order_idx][nocc_idx],
                [
                    ref_hash
                    for ref_hash, ref_tup in zip(
                        start_hashes[order_idx][nocc_idx],
                        start_tups[order_idx][nocc_idx][sort_idx],
                    )
                    if exp.screen_orbs[0] not in ref_tup
                ],
            )
            assert np.array_equal(
                purged_incs[order_idx][nocc_idx],
                [
                    ref_inc
                    for ref_inc, ref_tup in zip(
                        np.arange(
                            1,
                            start_tups[order_idx][nocc_idx].size + 1,
                            dtype=np.float64,
                        ),
                        start_tups[order_idx][nocc_idx][sort_idx],
                    )
                    if exp.screen_orbs[0] not in ref_tup
                ],
            )


@pytest.mark.parametrize(
    argnames="system, method, base_method, cc_backend, root, ref_res, ref_civec_sum, "
    "ref_civec_amax",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:4] if item]) for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: EnergyExpCls,
    ints: Tuple[np.ndarray, np.ndarray],
    vhf: np.ndarray,
    orbsym: np.ndarray,
    method: str,
    base_method: Optional[str],
    cc_backend: str,
    root: int,
    ref_res: float,
    ref_civec_sum: float,
    ref_civec_amax: float,
) -> None:
    """
    this function tests _ref_prop
    """
    exp.method = method
    exp.cc_backend = cc_backend
    exp.orbsym = orbsym
    exp.fci_state_root = root
    exp.hcore, exp.eri = ints
    exp.vhf = vhf
    exp.ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    exp.ref_nelec = np.array(
        [
            np.count_nonzero(exp.occup[exp.ref_space] > 0.0),
            np.count_nonzero(exp.occup[exp.ref_space] > 1.0),
        ],
    )
    exp.base_method = base_method

    res, civec = exp._ref_prop(mbe.mpi)

    assert res == pytest.approx(ref_res)
    assert np.sum(civec[0] ** 2) == pytest.approx(ref_civec_sum)
    assert np.amax(civec[0] ** 2) == pytest.approx(ref_civec_amax)


@pytest.mark.parametrize(
    argnames="system, method, cc_backend, ref_res",
    argvalues=test_cases_kernel,
    ids=["-".join([item for item in case[0:3] if item]) for case in test_cases_kernel],
    indirect=["system"],
)
def test_kernel(
    system: str,
    exp: EnergyExpCls,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    method: str,
    cc_backend: str,
    ref_res: float,
) -> None:
    """
    this function tests _kernel
    """
    exp.orbsym = orbsym

    if system == "h2o":
        exp.point_group = "C2v"
        occup = hf.mo_occ

    elif system == "hubbard":
        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
        exp.point_group = "C1"
        exp.hf_guess = False

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res = exp._kernel(
        method, 0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, ref_guess=False
    )

    assert res == pytest.approx(ref_res)


@pytest.mark.parametrize(
    argnames="system, ref, ref_civec_sum, ref_civec_amax",
    argvalues=test_cases_fci_kernel,
    ids=[case[0] for case in test_cases_fci_kernel],
    indirect=["system"],
)
def test_fci_kernel(
    exp: EnergyExpCls,
    system: str,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    ref: float,
    ref_civec_sum: float,
    ref_civec_amax: float,
) -> None:
    """
    this function tests _fci_kernel
    """
    exp.orbsym = orbsym

    if system == "h2o":
        occup = hf.mo_occ

    elif system == "hubbard":
        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
        exp.hf_guess = False

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res, civec = exp._fci_kernel(0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, False)

    assert res == pytest.approx(ref)
    assert np.sum(civec[0] ** 2) == pytest.approx(ref_civec_sum)
    assert np.amax(civec[0] ** 2) == pytest.approx(ref_civec_amax)


@pytest.mark.parametrize(
    argnames="system, method, cc_backend, ref",
    argvalues=test_cases_cc_kernel,
    ids=["-".join(case[0:3]) for case in test_cases_cc_kernel],
    indirect=["system"],
)
def test_cc_kernel(
    exp: EnergyExpCls,
    hf: scf.RHF,
    orbsym: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    method: str,
    cc_backend: str,
    ref: float,
) -> None:
    """
    this function tests _cc_kernel
    """
    exp.cc_backend = cc_backend
    exp.point_group = "C2v"
    exp.orbsym = orbsym

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(hf.mo_occ[cas_idx] > 0.0),
            np.count_nonzero(hf.mo_occ[cas_idx] > 1.0),
        ]
    )

    res = exp._cc_kernel(method, core_idx, cas_idx, nelec, h1e_cas, h2e_cas, False)

    assert res == pytest.approx(ref)
