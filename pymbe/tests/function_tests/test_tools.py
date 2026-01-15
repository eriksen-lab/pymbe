#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools testing module
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
from typing import TYPE_CHECKING
from types import GeneratorType

from pymbe.tools import (
    time_str,
    hash_1d,
    hash_lookup,
    tuples,
    _start_idx,
    _comb_idx,
    _idx,
    n_tuples,
    n_tuples_with_nocc,
    cas,
    core_cas,
    _cas_idx_cart,
    _coor_to_idx,
    idx_tril,
    get_nelec,
    get_nhole,
    valid_tup,
    natural_keys,
    _convert,
    intervals,
    ground_state_sym,
    get_vhf,
    get_occup,
    e_core_h1e,
)

if TYPE_CHECKING:
    from typing import Union, Tuple, List, Optional


test_cases_hash_lookup = [
    ("present", np.array([1, 3, 5, 7, 9], dtype=np.int64), True),
    ("absent", np.array([1, 3, 5, 7, 11], dtype=np.int64), False),
]

test_cases_tuples = [
    ("no_ref_space", np.array([], dtype=np.int64), 18),
    ("ref_space", np.array([3, 4], dtype=np.int64), 20),
]

test_cases_start_idx = [
    (
        "all",
        np.array([1, 2], dtype=np.int64),
        np.array([6, 7, 12], dtype=np.int64),
        (2, 3, 1),
    ),
    (
        "occ",
        np.array([0, 1, 2], dtype=np.int64),
        None,
        (3, 0, 0),
    ),
    (
        "virt",
        None,
        np.array([6, 9, 12], dtype=np.int64),
        (0, 0, 2),
    ),
]

test_cases_comb_idx = [
    (np.array([1, 2, 6, 7], dtype=np.int64), 12.0),
    (np.array([1, 2], dtype=np.int64), 5.0),
    (np.array([5, 7], dtype=np.int64), 13.0),
]

test_cases_idx = [(1, 3.0), (2, 12.0), (3, 19.0)]

test_cases_n_tuples = [
    ("both", np.array([1, 1]), np.array([1, 1]), 2118760),
    ("no-occ", np.array([0, 0]), np.array([1, 1]), 1460752),
    ("no-virt", np.array([1, 1]), np.array([0, 0]), 2118508),
    ("empty", np.array([0, 0]), np.array([0, 0]), 1460500),
]

test_cases_valid_tup = [
    ("none-occ-1", np.array([0, 0]), np.array([0, 0]), 1, 0, 1, False),
    ("none-virt-1", np.array([0, 0]), np.array([0, 0]), 0, 1, 1, False),
    ("none-both-1", np.array([0, 0]), np.array([0, 0]), 1, 1, 1, True),
    ("occ-occ-1", np.array([1, 1]), np.array([0, 0]), 1, 0, 1, False),
    ("occ-virt-1", np.array([1, 1]), np.array([0, 0]), 0, 1, 1, True),
    ("occ-both-1", np.array([1, 1]), np.array([0, 0]), 1, 1, 1, True),
    ("virt-occ-1", np.array([0, 0]), np.array([1, 1]), 1, 0, 1, True),
    ("virt-virt-1", np.array([0, 0]), np.array([1, 1]), 0, 1, 1, False),
    ("virt-both-1", np.array([0, 0]), np.array([1, 1]), 1, 1, 1, True),
    ("both-occ-1", np.array([1, 1]), np.array([1, 1]), 1, 0, 1, True),
    ("both-virt-1", np.array([1, 1]), np.array([1, 1]), 0, 1, 1, True),
    ("both-both-1", np.array([1, 1]), np.array([1, 1]), 1, 1, 1, True),
    ("none-occ-2", np.array([0, 0]), np.array([0, 0]), 1, 0, 2, False),
    ("none-virt-2", np.array([0, 0]), np.array([0, 0]), 0, 1, 2, False),
    ("none-both-2", np.array([0, 0]), np.array([0, 0]), 1, 1, 2, False),
    ("occ-occ-2", np.array([1, 1]), np.array([0, 0]), 1, 0, 2, False),
    ("occ-virt-2", np.array([1, 1]), np.array([0, 0]), 0, 1, 2, False),
    ("occ-both-2", np.array([1, 1]), np.array([0, 0]), 1, 1, 2, False),
    ("virt-occ-2", np.array([0, 0]), np.array([1, 1]), 1, 0, 2, False),
    ("virt-virt-2", np.array([0, 0]), np.array([1, 1]), 0, 1, 2, False),
    ("virt-both-2", np.array([0, 0]), np.array([1, 1]), 1, 1, 2, False),
    ("both-occ-2", np.array([1, 1]), np.array([1, 1]), 1, 0, 2, False),
    ("both-virt-2", np.array([1, 1]), np.array([1, 1]), 0, 1, 2, False),
    ("both-both-2", np.array([1, 1]), np.array([1, 1]), 1, 1, 2, True),
]

test_cases_get_nelec = [
    ("4_elecs", np.array([1, 2], dtype=np.int64), np.array([2, 2])),
    ("2_elecs", np.array([2, 4], dtype=np.int64), np.array([1, 1])),
    ("no_elecs", np.array([3, 4], dtype=np.int64), np.array([0, 0])),
]

test_cases_get_nhole = [
    ("4_holes", np.array([0, 0]), np.array([3, 4], dtype=np.int64), np.array([2, 2])),
    ("2_holes", np.array([1, 1]), np.array([2, 4], dtype=np.int64), np.array([1, 1])),
    ("no_holes", np.array([2, 2]), np.array([3, 4], dtype=np.int64), np.array([0, 0])),
]

test_cases_natural_keys = [
    ("str", "mbe_test_string", ["mbe_test_string"]),
    ("str+int", "mbe_test_string_1", ["mbe_test_string_", 1, ""]),
]

test_cases_convert = [("str", "string", str), ("int", "1", int)]

test_cases_ground_state_sym = [
    (
        "closed-shell",
        np.arange(0, 4, dtype=np.int64),
        np.array([2, 2], dtype=np.int64),
        "c2v",
        0,
    ),
    (
        "open-shell",
        np.arange(0, 4, dtype=np.int64),
        np.array([3, 1], dtype=np.int64),
        "c2v",
        3,
    ),
]

test_cases_get_occup = [
    (
        "closed-shell",
        4,
        np.array([2, 2], dtype=np.int64),
        np.array([2, 2, 0, 0], dtype=np.int64),
    ),
    (
        "open-shell",
        4,
        np.array([3, 1], dtype=np.int64),
        np.array([2, 1, 1, 0], dtype=np.int64),
    ),
]


def test_time_str() -> None:
    """
    this function tests time_str
    """
    assert time_str(3742.4) == "1h 2m 22.40s"


def test_hash_1d() -> None:
    """
    this function tests hash_1d
    """
    arr1 = np.arange(5, dtype=np.int64)
    arr2 = np.array([0, 1, 2, 3, 5], dtype=np.int64)

    assert hash_1d(arr1) == hash(arr1.tobytes())
    assert hash_1d(arr1) != hash_1d(arr2)


@pytest.mark.parametrize(
    argnames="b, present",
    argvalues=[case[1:] for case in test_cases_hash_lookup],
    ids=[case[0] for case in test_cases_hash_lookup],
)
def test_hash_lookup(b: np.ndarray, present: bool) -> None:
    """
    this function tests hash_lookup
    """
    a = np.arange(10, dtype=np.int64)

    if present:
        assert (hash_lookup(a, b) == b).all()
    else:
        assert hash_lookup(a, b) is None


@pytest.mark.parametrize(
    argnames="ref_space, ref_n_tuples",
    argvalues=[case[1:] for case in test_cases_tuples],
    ids=[case[0] for case in test_cases_tuples],
)
def test_tuples(ref_space: np.ndarray, ref_n_tuples: int) -> None:
    """
    this function tests tuples
    """
    nocc = 4
    order = 3
    occup = np.array([2.0] * 4 + [0.0] * 4, dtype=np.float64)
    exp_space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)
    ref_nelec = get_nelec(occup, ref_space)
    ref_nhole = get_nhole(ref_nelec, ref_space)

    gen = tuples(exp_space, None, nocc, ref_nelec, ref_nhole, 1, order)

    assert isinstance(gen, GeneratorType)
    assert sum(1 for _ in gen) == ref_n_tuples


@pytest.mark.parametrize(
    argnames="tup_occ, tup_virt, ref_start_idx",
    argvalues=[case[1:] for case in test_cases_start_idx],
    ids=[case[0] for case in test_cases_start_idx],
)
def test_start_idx(
    tup_occ: Optional[np.ndarray],
    tup_virt: Optional[np.ndarray],
    ref_start_idx: Tuple[int, int, int],
) -> None:
    """
    this function tests _start_idx
    """
    occ_space = np.array([0, 1, 2, 5], dtype=np.int64)
    virt_space = np.array([6, 7, 9, 12], dtype=np.int64)

    assert _start_idx(occ_space, virt_space, tup_occ, tup_virt) == ref_start_idx


@pytest.mark.parametrize(
    argnames="tup, ref_idx",
    argvalues=test_cases_comb_idx,
    ids=[str(case[0]) for case in test_cases_comb_idx],
)
def test_comb_idx(tup: np.ndarray, ref_idx: float) -> None:
    """
    this function tests _comb_idx
    """
    space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)

    assert _comb_idx(space, tup) == ref_idx


@pytest.mark.parametrize(
    argnames="order, ref_idx",
    argvalues=test_cases_idx,
    ids=[case[0] for case in test_cases_idx],
)
def test_idx(order: int, ref_idx: float) -> None:
    """
    this function tests _idx
    """
    space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)

    assert _idx(space, 5, order) == ref_idx


@pytest.mark.parametrize(
    argnames="ref_nelec, ref_nhole, ref_n_tuples",
    argvalues=[case[1:] for case in test_cases_n_tuples],
    ids=[case[0] for case in test_cases_n_tuples],
)
def test_n_tuples(
    ref_nelec: np.ndarray, ref_nhole: np.ndarray, ref_n_tuples: int
) -> None:
    """
    this function tests n_tuples
    """
    nocc = 10
    order = 5
    exp_space = np.arange(50, dtype=np.int64)

    assert (
        n_tuples(exp_space, None, nocc, ref_nelec, ref_nhole, 1, order) == ref_n_tuples
    )


@pytest.mark.parametrize(
    argnames="ref_nelec, ref_nhole, ref_n_tuples",
    argvalues=[case[1:] for case in test_cases_n_tuples],
    ids=[case[0] for case in test_cases_n_tuples],
)
def test_n_tuples_with_nocc(
    ref_nelec: np.ndarray, ref_nhole: np.ndarray, ref_n_tuples: int
) -> None:
    """
    this function tests n_tuples
    """
    nocc = 10
    order = 5
    exp_space = np.arange(50, dtype=np.int64)

    ntuples = 0
    for tup_nocc in range(order + 1):
        ntuples += n_tuples_with_nocc(
            exp_space, None, nocc, ref_nelec, ref_nhole, 1, order, tup_nocc
        )

    assert ntuples == ref_n_tuples


def test_cas() -> None:
    """
    this function tests cas
    """
    assert (
        cas(np.array([7, 13], dtype=np.int64), np.arange(5, dtype=np.int64))
        == np.array([0, 1, 2, 3, 4, 7, 13], dtype=np.int64)
    ).all()


def test_core_cas() -> None:
    """
    this function tests core_cas
    """
    core_idx, cas_idx = core_cas(
        8, np.arange(3, 5, dtype=np.int64), np.array([9, 21], dtype=np.int64)
    )

    assert (core_idx == np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)).all()
    assert (cas_idx == np.array([3, 4, 9, 21], dtype=np.int64)).all()


def test_cas_idx_cart() -> None:
    """
    this function tests _cas_idx_cart
    """
    ref = np.array(
        [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]],
        dtype=np.int64,
    )

    assert (_cas_idx_cart(np.arange(0, 7, 3, dtype=np.int64)) == ref).all()


def test_coor_to_idx() -> None:
    """
    this function tests _coor_to_idx
    """
    assert _coor_to_idx((4, 9)) == 49


def test_idx_tril() -> None:
    """
    this function tests idx_tril
    """
    ref = np.array([5, 17, 20, 38, 41, 44, 68, 71, 74, 77], dtype=np.int64)

    assert (idx_tril(np.arange(2, 14, 3, dtype=np.int64)) == ref).all()


@pytest.mark.parametrize(
    argnames="ref_nelec, ref_nhole, tup_nocc, tup_nvirt, vanish_exc, ref_bool",
    argvalues=[case[1:] for case in test_cases_valid_tup],
    ids=[case[0] for case in test_cases_valid_tup],
)
def test_valid_tup(
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    tup_nocc: int,
    tup_nvirt: int,
    vanish_exc: int,
    ref_bool: bool,
) -> None:
    """
    this function tests _valid_tup
    """
    assert valid_tup(ref_nelec, ref_nhole, tup_nocc, tup_nvirt, vanish_exc) == ref_bool


@pytest.mark.parametrize(
    argnames="tup, ref_get_nelec",
    argvalues=[case[1:] for case in test_cases_get_nelec],
    ids=[case[0] for case in test_cases_get_nelec],
)
def test_get_nelec(tup: np.ndarray, ref_get_nelec: np.ndarray) -> None:
    """
    this function tests nelec
    """
    occup = np.array([2.0] * 3 + [0.0] * 4, dtype=np.float64)

    assert (get_nelec(occup, tup) == ref_get_nelec).all()


@pytest.mark.parametrize(
    argnames="nelec, tup, ref_get_nhole",
    argvalues=[case[1:] for case in test_cases_get_nhole],
    ids=[case[0] for case in test_cases_get_nhole],
)
def test_get_nhole(
    nelec: np.ndarray, tup: np.ndarray, ref_get_nhole: np.ndarray
) -> None:
    """
    this function tests get_nhole
    """
    assert (get_nhole(nelec, tup) == ref_get_nhole).all()


@pytest.mark.parametrize(
    argnames="test_string, ref_keys",
    argvalues=[case[1:] for case in test_cases_natural_keys],
    ids=[case[0] for case in test_cases_natural_keys],
)
def test_natural_keys(test_string: str, ref_keys: List[Union[int, str]]) -> None:
    """
    this function tests natural_keys
    """
    assert natural_keys(test_string) == ref_keys


@pytest.mark.parametrize(
    argnames="test_string, ref_type",
    argvalues=[case[1:] for case in test_cases_convert],
    ids=[case[0] for case in test_cases_convert],
)
def test_convert(test_string: str, ref_type: type) -> None:
    """
    this function tests _convert
    """
    assert isinstance(_convert(test_string), ref_type)


def test_intervals() -> None:
    """
    this function tests intervals
    """
    ref = [[0, 2], [4], [6, 7]]
    assert [i for i in intervals(np.array([0, 1, 2, 4, 6, 7], dtype=np.int64))] == ref


@pytest.mark.parametrize(
    argnames="orbsym, occup, point_group, ref_wfnsym",
    argvalues=[case[1:] for case in test_cases_ground_state_sym],
    ids=[case[0] for case in test_cases_ground_state_sym],
)
def test_ground_state_sym(
    orbsym: np.ndarray, occup: np.ndarray, point_group: str, ref_wfnsym: int
) -> None:
    """
    this function tests ground_state_sym
    """
    assert ground_state_sym(orbsym, occup, point_group) == ref_wfnsym


def test_get_vhf() -> None:
    """
    this function tests get_vhf
    """
    vhf = get_vhf(np.arange(55, dtype=np.float64), 2, 4)

    assert (
        vhf
        == np.array(
            [
                [
                    [0.0, 1.0, 6.0, 21.0],
                    [1.0, 4.0, 13.0, 34.0],
                    [6.0, 13.0, 21.0, 48.0],
                    [21.0, 34.0, 48.0, 63.0],
                ],
                [
                    [4.0, 4.0, 5.0, 17.0],
                    [4.0, 5.0, 12.0, 30.0],
                    [5.0, 12.0, 20.0, 44.0],
                    [17.0, 30.0, 44.0, 59.0],
                ],
            ]
        )
    ).all()


@pytest.mark.parametrize(
    argnames="norb, nelec, ref_occup",
    argvalues=[case[1:] for case in test_cases_get_occup],
    ids=[case[0] for case in test_cases_get_occup],
)
def test_get_occup(norb: int, nelec: np.ndarray, ref_occup: np.ndarray) -> None:
    """
    this function tests get_occup
    """
    assert (get_occup(norb, nelec) == ref_occup).all()


def test_e_core_h1e() -> None:
    """
    this function tests e_core_h1e
    """
    np.random.seed(1234)
    hcore = np.random.rand(6, 6)
    np.random.seed(1234)
    vhf = np.random.rand(3, 6, 6)
    core_idx = np.array([0], dtype=np.int64)
    cas_idx = np.array([2, 4, 5], dtype=np.int64)
    e_core, h1e_cas = e_core_h1e(hcore, vhf, core_idx, cas_idx)

    assert e_core == pytest.approx(0.5745583511366769)
    assert h1e_cas == pytest.approx(
        np.array(
            [
                [0.74050151, 1.00616633, 0.02753690],
                [0.79440516, 0.63367224, 1.13619731],
                [1.60429528, 1.40852194, 1.40916262],
            ],
            dtype=np.float64,
        )
    )
