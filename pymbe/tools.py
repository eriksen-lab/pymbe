#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import sys
import re
import operator
import functools
import numpy as np
from math import comb
from pyscf import symm, ao2mo, fci
from itertools import islice, combinations, groupby, chain, product
from bisect import insort
from subprocess import Popen, PIPE
from typing import TYPE_CHECKING, overload, TypeVar, List

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from pymbe.logger import logger

if TYPE_CHECKING:
    from typing import Tuple, Generator, Union, Optional, Dict, Set, Callable


# Generic type
T = TypeVar("T")


# Type for dictionary holding tuples and their squared overlap values
TupSqOverlapType = TypedDict(
    "TupSqOverlapType", {"overlap": List[float], "tup": List[np.ndarray]}
)


# restart folder
RST = os.getcwd() + "/rst"

# ids for doubly degenerate irreducible representations in linear point groups
E_IRREPS = np.array(
    [2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27]
)


class RDMCls:
    """
    this class holds the 1- and 2-particle RDMs and defines all necessary operations
    """

    def __init__(self, rdm1: np.ndarray, rdm2: np.ndarray):
        """
        initializes a RDM object
        """
        self.rdm1 = rdm1
        self.rdm2 = rdm2

    def __getitem__(self, idx: np.ndarray) -> RDMCls:
        """
        this function ensures RDMCls can be retrieved using one-dimensional indexing of
        RDMCls objects
        """
        return RDMCls(
            self.rdm1[idx.reshape(-1, 1), idx],
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ],
        )

    def __setitem__(
        self,
        idx: np.ndarray,
        values: Union[RDMCls, Tuple[np.ndarray, np.ndarray]],
    ) -> RDMCls:
        """
        this function implements setting RDMs through indexing for the RDMCls objects
        this function ensures RDMCls indexed in one dimension can be set using RDMCls
        or tuples of numpy arrays
        """
        if isinstance(values, RDMCls):
            self.rdm1[idx.reshape(-1, 1), idx] = values.rdm1
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ] = values.rdm2
        elif isinstance(values, tuple):
            self.rdm1[idx.reshape(-1, 1), idx] = values[0]
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ] = values[1]
        else:
            return NotImplemented

        return self

    def __add__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements addition for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            return RDMCls(self.rdm1 + other.rdm1, self.rdm2 + other.rdm2)
        else:
            return NotImplemented

    def __iadd__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements inplace addition for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            self.rdm1 += other.rdm1
            self.rdm2 += other.rdm2
            return self
        else:
            return NotImplemented

    def __sub__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements subtraction for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            return RDMCls(self.rdm1 - other.rdm1, self.rdm2 - other.rdm2)
        else:
            return NotImplemented

    def __isub__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements inplace subtraction for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            self.rdm1 -= other.rdm1
            self.rdm2 -= other.rdm2
            return self
        else:
            return NotImplemented

    def __mul__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements multiplication for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            return RDMCls(other * self.rdm1, other * self.rdm2)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements inplace multiplication for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            self.rdm1 *= other
            self.rdm2 *= other
            return self
        else:
            return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements division for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            return RDMCls(self.rdm1 / other, self.rdm2 / other)
        else:
            return NotImplemented

    __rtruediv__ = __truediv__

    def __itruediv__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements inplace division for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            self.rdm1 /= other
            self.rdm2 /= other
            return self
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for RDMCls objects
        """
        self.rdm1.fill(value)
        self.rdm2.fill(value)

    def copy(self) -> RDMCls:
        """
        this function creates a copy of RDMCls objects
        """
        return RDMCls(self.rdm1.copy(), self.rdm2.copy())


class packedRDMCls:
    """
    this class describes packed RDMs
    """

    rdm1_size: List[int] = []
    pack_rdm1: List[Tuple[np.ndarray, np.ndarray]] = []
    unpack_rdm1: List[np.ndarray] = []
    rdm2_size: List[int] = []
    pack_rdm2: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    unpack_rdm2: List[np.ndarray] = []

    def __init__(self, rdm1: np.ndarray, rdm2: np.ndarray, idx: int = -1) -> None:
        """
        this function initializes a packedRDMCls object
        """
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.idx = idx

    @classmethod
    def reset(cls):
        """
        this function resets the class to ensure class attributes are not kept from
        previous runs
        """
        cls.rdm1_size = []
        cls.pack_rdm1 = []
        cls.unpack_rdm1 = []
        cls.rdm2_size = []
        cls.pack_rdm2 = []
        cls.unpack_rdm2 = []

    @classmethod
    def get_pack_idx(cls, norb) -> None:
        """
        this function generates packing and unpacking indices for the 1- and 2-particle
        rdms and should be called at every order before a packedRDMCls object is
        initialized at this order
        """
        pack_rdm1 = np.triu_indices(norb)
        unpack_rdm1 = np.zeros((norb, norb), dtype=np.int64)
        rdm1_size = pack_rdm1[0].size
        indices = np.arange(rdm1_size)
        unpack_rdm1[pack_rdm1] = indices
        unpack_rdm1[pack_rdm1[1], pack_rdm1[0]] = indices

        rdm2_size = (7 * norb**4 + 8 * norb**3 - norb**2 - 2 * norb) // 12
        pack_rdm2 = np.empty((rdm2_size, 4), dtype=np.int64)
        unpack_rdm2 = np.empty(4 * (norb,), dtype=np.int64)

        i = 0
        for s in range(norb):
            for r in range(s + 1):
                for q in range(s + 1):
                    for p in range(norb):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
                for q in range(s + 1, norb):
                    for p in range(r + 1):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
            for r in range(s + 1, norb):
                for q in range(s + 1):
                    for p in range(q + 1):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
                for q in range(s + 1, norb):
                    for p in range(min(r + 1, q + 1)):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1

        cls.rdm1_size.append(rdm1_size)
        cls.pack_rdm1.append(pack_rdm1)
        cls.unpack_rdm1.append(unpack_rdm1)
        cls.rdm2_size.append(rdm2_size)
        cls.pack_rdm2.append(
            (pack_rdm2[:, 0], pack_rdm2[:, 1], pack_rdm2[:, 2], pack_rdm2[:, 3])
        )
        cls.unpack_rdm2.append(unpack_rdm2)

    @overload
    def __getitem__(self, idx: Union[int, np.int64]) -> RDMCls: ...

    @overload
    def __getitem__(self, idx: Union[slice, np.ndarray, List[int]]) -> packedRDMCls: ...

    def __getitem__(
        self, idx: Union[int, np.int64, slice, np.ndarray, List[int]]
    ) -> Union[RDMCls, packedRDMCls]:
        """
        this function ensures packedRDMCls can be retrieved through indexing
        packedRDMCls objects
        """
        if isinstance(idx, (int, np.integer)):
            return RDMCls(
                self.rdm1[idx][self.unpack_rdm1[self.idx]],
                self.rdm2[idx][self.unpack_rdm2[self.idx]],
            )
        elif isinstance(idx, (slice, np.ndarray, list)):
            return packedRDMCls(self.rdm1[idx], self.rdm2[idx], self.idx)
        else:
            return NotImplemented

    def __setitem__(
        self,
        idx: Union[int, np.int64, slice, np.ndarray, list],
        values: Union[
            float, np.ndarray, RDMCls, packedRDMCls, GenFockCls, packedGenFockCls
        ],
    ) -> packedRDMCls:
        """
        this function ensures indexed packedRDMCls can be set using packedRDMCls or
        RDMCls objects
        """
        if isinstance(idx, (slice, np.ndarray, list)) and isinstance(
            values, packedRDMCls
        ):
            self.rdm1[idx] = values.rdm1
            self.rdm2[idx] = values.rdm2
        elif isinstance(idx, (int, np.integer)) and isinstance(values, RDMCls):
            self.rdm1[idx] = values.rdm1[self.pack_rdm1[self.idx]]
            self.rdm2[idx] = values.rdm2[self.pack_rdm2[self.idx]]
        else:
            return NotImplemented

        return self

    def __radd__(self, other: RDMCls) -> RDMCls:
        """
        this function ensures the packedRDMCls object is unpacked when added to a RDMCls
        object
        """
        if isinstance(other, RDMCls) and self.rdm1.ndim == 1 and self.rdm2.ndim == 1:
            return other + RDMCls(
                self.rdm1[self.unpack_rdm1[self.idx]],
                self.rdm2[self.unpack_rdm2[self.idx]],
            )
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for packedRDMCls objects
        """
        self.rdm1.fill(value)
        self.rdm2.fill(value)


class GenFockCls:
    """
    this class holds the energy and generalized Fock matrix elements and defines all
    necessary operations
    """

    def __init__(self, energy: float, rdm1: np.ndarray, gen_fock: np.ndarray):
        """
        initializes a GenFock object
        """
        self.energy = energy
        self.rdm1 = rdm1
        self.gen_fock = gen_fock

    def __getitem__(
        self,
        idx: Tuple[np.ndarray, np.ndarray],
    ) -> GenFockCls:
        """
        this function ensures GenFockCls can be retrieved using one-dimensional
        indexing of GenFockCls objects
        """
        return GenFockCls(
            self.energy, self.rdm1[idx[0].reshape(-1, 1), idx[0]], self.gen_fock[idx[1]]
        )

    def __setitem__(
        self,
        idx: Tuple[np.ndarray, np.ndarray],
        values: Union[GenFockCls, Tuple[float, np.ndarray, np.ndarray]],
    ) -> GenFockCls:
        """
        this function implements setting GenFockCls through indexing for the GenFockCls
        objects and through tuples of a float and two arrays
        """
        if isinstance(values, GenFockCls):
            self.energy = values.energy
            self.rdm1[idx[0].reshape(-1, 1), idx[0]] = values.rdm1
            self.gen_fock[idx[1]] = values.gen_fock
        elif isinstance(values, tuple):
            self.energy = values[0]
            self.rdm1[idx[0].reshape(-1, 1), idx[0]] = values[1]
            self.gen_fock[idx[1]] = values[2]
        else:
            return NotImplemented

        return self

    def __add__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements addition for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            return GenFockCls(
                self.energy + other.energy,
                self.rdm1 + other.rdm1,
                self.gen_fock + other.gen_fock,
            )
        else:
            return NotImplemented

    def __iadd__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements inplace addition for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            self.energy += other.energy
            self.rdm1 += other.rdm1
            self.gen_fock += other.gen_fock
            return self
        else:
            return NotImplemented

    def __sub__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements subtraction for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            return GenFockCls(
                self.energy - other.energy,
                self.rdm1 - other.rdm1,
                self.gen_fock - other.gen_fock,
            )
        else:
            return NotImplemented

    def __isub__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements inplace subtraction for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            self.energy -= other.energy
            self.rdm1 -= other.rdm1
            self.gen_fock -= other.gen_fock
            return self
        else:
            return NotImplemented

    def __mul__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements multiplication for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            return GenFockCls(
                other * self.energy, other * self.rdm1, other * self.gen_fock
            )
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements inplace multiplication for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            self.energy *= other
            self.rdm1 *= other
            self.gen_fock *= other
            return self
        else:
            return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements division for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            return GenFockCls(
                self.energy / other, self.rdm1 / other, self.gen_fock / other
            )
        else:
            return NotImplemented

    def __itruediv__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements inplace division for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            self.energy /= other
            self.rdm1 /= other
            self.gen_fock /= other
            return self
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for GenFockCls objects
        """
        self.energy = value
        self.rdm1.fill(value)
        self.gen_fock.fill(value)

    def copy(self) -> GenFockCls:
        """
        this function creates a copy of GenFockCls objects
        """
        return GenFockCls(self.energy, self.rdm1.copy(), self.gen_fock.copy())


class packedGenFockCls:
    """
    this class describes a packed version of GenFockCls
    """

    rdm1_size: List[int] = []
    pack_rdm1: List[Tuple[np.ndarray, np.ndarray]] = []
    unpack_rdm1: List[np.ndarray] = []

    def __init__(
        self, energy: np.ndarray, rdm1: np.ndarray, gen_fock: np.ndarray, idx: int = -1
    ) -> None:
        """
        this function initializes a packedGenFockCls object
        """
        self.energy = energy
        self.rdm1 = rdm1
        self.gen_fock = gen_fock
        self.idx = idx

    @classmethod
    def reset(cls):
        """
        this function resets the class to ensure class attributes are not kept from
        previous runs
        """
        cls.rdm1_size = []
        cls.pack_rdm1 = []
        cls.unpack_rdm1 = []

    @classmethod
    def get_pack_idx(cls, norb) -> None:
        """
        this function generates packing and unpacking indices for the 1-particle rdms
        and should be called at every order before a packedGenFockCls object is
        initialized at this order
        """
        pack_rdm1 = np.triu_indices(norb)
        unpack_rdm1 = np.zeros((norb, norb), dtype=np.int64)
        rdm1_size = pack_rdm1[0].size
        indices = np.arange(rdm1_size)
        unpack_rdm1[pack_rdm1] = indices
        unpack_rdm1[pack_rdm1[1], pack_rdm1[0]] = indices

        cls.rdm1_size.append(rdm1_size)
        cls.pack_rdm1.append(pack_rdm1)
        cls.unpack_rdm1.append(unpack_rdm1)

    @overload
    def __getitem__(self, idx: Union[int, np.int64]) -> GenFockCls: ...

    @overload
    def __getitem__(
        self, idx: Union[slice, np.ndarray, List[int]]
    ) -> packedGenFockCls: ...

    def __getitem__(
        self, idx: Union[int, np.int64, slice, np.ndarray, List[int]]
    ) -> Union[GenFockCls, packedGenFockCls]:
        """
        this function ensures packedGenFockCls can be retrieved through indexing
        packedGenFockCls objects
        """
        if isinstance(idx, (int, np.int64, slice, np.ndarray, list)):
            return packedGenFockCls(
                self.energy[idx], self.rdm1[idx], self.gen_fock[idx], self.idx
            )
        else:
            return NotImplemented

    def __setitem__(
        self,
        idx: Union[int, np.int64, slice, np.ndarray, List[int]],
        values: Union[
            float, np.ndarray, RDMCls, packedRDMCls, GenFockCls, packedGenFockCls
        ],
    ) -> packedGenFockCls:
        """
        this function ensures indexed packedGenFockCls can be set using packedGenFockCls
        or GenFockCls objects
        """
        if isinstance(idx, (slice, np.ndarray, list)) and isinstance(
            values, packedGenFockCls
        ):
            self.energy[idx] = values.energy
            self.rdm1[idx] = values.rdm1
            self.gen_fock[idx] = values.gen_fock
        elif isinstance(idx, (int, np.integer)) and isinstance(values, GenFockCls):
            self.energy[idx] = values.energy
            self.rdm1[idx] = values.rdm1[self.pack_rdm1[self.idx]]
            self.gen_fock[idx] = values.gen_fock
        else:
            return NotImplemented

        return self

    def __radd__(self, other: GenFockCls) -> GenFockCls:
        """
        this function ensures the packedGenFockCls object is unpacked when added to a
        GenFockCls object
        """
        if (
            isinstance(other, GenFockCls)
            and self.energy.size == 1
            and self.rdm1.ndim == 1
            and self.gen_fock.ndim == 2
        ):
            return other + GenFockCls(
                self.energy.item(), self.rdm1[self.unpack_rdm1[self.idx]], self.gen_fock
            )
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for packedGenFockCls objects
        """
        self.energy.fill(value)
        self.rdm1.fill(value)
        self.gen_fock.fill(value)


def logger_config(verbose: int) -> None:
    """
    this function configures the pymbe logger
    """
    # corresponding logging level
    verbose_level = {0: 30, 1: 20, 2: 19, 3: 18, 4: 10, 5: 5}

    # set level for logger
    logger.setLevel(verbose_level[verbose])


def git_version() -> str:
    """
    this function returns the git revision as a string
    see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92
    """

    def _minimal_ext_cmd(cmd: List[str]) -> bytes:
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = Popen(cmd, stdout=PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_pymbe_path() -> str:
    """
    this function returns the path to pymbe
    """
    return os.path.dirname(__file__)


def time_str(time: float) -> str:
    """
    this function returns time as a HH:MM:SS string
    """
    # hours, minutes, and seconds
    hours = time // 3600.0
    minutes = (time - (time // 3600) * 3600.0) // 60.0
    seconds = time - hours * 3600.0 - minutes * 60.0

    # init time string
    string: str = ""

    # write time string
    if hours > 0:
        string += f"{hours:.0f}h "

    if minutes > 0:
        string += f"{minutes:.0f}m "

    string += f"{seconds:.2f}s"

    return string


def hash_1d(a: np.ndarray) -> int:
    """
    this function converts a 1d numpy array to a hash
    """
    return hash(a.tobytes())


def hash_lookup(a: np.ndarray, b: Union[int, np.ndarray]) -> Optional[np.ndarray]:
    """
    this function finds occurences of b in a through a binary search
    """
    left = a.searchsorted(b, side="left")
    right = a.searchsorted(b, side="right")
    if ((right - left) > 0).all():
        return left
    else:
        return None


def tuples(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    start_tup: Optional[np.ndarray] = None,
    start_idx: int = 0,
) -> Generator[Tuple[np.ndarray, Optional[List[np.ndarray]]], None, None]:
    """
    this function is the main generator for tuples
    """
    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # start tuples
        start_tup_occ: Optional[np.ndarray]
        start_tup_virt: Optional[np.ndarray]
        if start_tup is not None:
            start_tup_occ = start_tup[start_tup < nocc]
            start_tup_virt = start_tup[nocc <= start_tup]
            if start_tup_occ.size == 0:
                start_tup_occ = None
            if start_tup_virt.size == 0:
                start_tup_virt = None
        else:
            start_tup_occ = start_tup_virt = None
        nocc_start, occ_start, virt_start = _start_idx(
            occ_space, virt_space, start_tup_occ, start_tup_virt
        )

        # loop over number of occupied orbitals
        for tup_nocc in range(nocc_start, order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                if tup_nocc == 0:
                    # only virtual MOs
                    for tup_virt in islice(
                        combinations(virt_space, order), virt_start, None
                    ):
                        yield np.array(tup_virt, dtype=np.int64), None
                    virt_start = 0
                elif 0 < tup_nocc < order:
                    # combinations of occupied and virtual MOs
                    for tup_occ in islice(
                        combinations(occ_space, tup_nocc), occ_start, None
                    ):
                        for tup_virt in islice(
                            combinations(virt_space, order - tup_nocc), virt_start, None
                        ):
                            yield np.array(tup_occ + tup_virt, dtype=np.int64), None
                        virt_start = 0
                    occ_start = 0
                elif tup_nocc == order:
                    # only occupied MOs
                    for tup_occ in islice(
                        combinations(occ_space, order), occ_start, None
                    ):
                        yield np.array(tup_occ, dtype=np.int64), None
                    occ_start = 0

    else:
        # determine start_idx by counting number of tuples per nocc before this
        if start_idx > 0:
            size_nvirt_blks = get_ncluster_blks(orb_clusters, nocc)
            for nocc_start in range(0, order + 1):
                if valid_tup(
                    ref_nelec, ref_nhole, nocc_start, order - nocc_start, vanish_exc
                ):
                    idx = _recursive_ntuples(
                        size_nvirt_blks, 0, 1, order, order - nocc_start
                    )
                    if idx < start_idx:
                        start_idx -= idx
                    else:
                        break
        else:
            nocc_start = 0

        # get number of clusters for every cluster size and number of virtual orbitals
        cluster_info = _cluster_info(orb_clusters, nocc)

        # loop over number of occupied orbitals in tuple
        for tup_nocc in range(nocc_start, order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                # loop over tuples
                for tup in islice(
                    _recursive_tuples(
                        cluster_info, orb_clusters, order, order - tup_nocc, 0, 0, 0
                    ),
                    start_idx,
                    None,
                ):
                    tup_arr = np.concatenate(tup)
                    tup_arr.sort()
                    yield tup_arr, tup
            # reset start index
            start_idx = 0


def _start_idx(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    tup_occ: Optional[np.ndarray],
    tup_virt: Optional[np.ndarray],
) -> Tuple[int, int, int]:
    """
    this function return the start indices for a given occupied and virtual tuple
    """
    if tup_occ is None and tup_virt is None:
        order_start = 0
        occ_start = virt_start = 0
    elif tup_occ is not None and tup_virt is not None:
        order_start = tup_occ.size
        occ_start = _comb_idx(occ_space, tup_occ)
        virt_start = _comb_idx(virt_space, tup_virt)
    elif tup_occ is not None and tup_virt is None:
        order_start = tup_occ.size
        occ_start = _comb_idx(occ_space, tup_occ)
        virt_start = 0
    elif tup_occ is None and tup_virt is not None:
        order_start = 0
        occ_start = 0
        virt_start = _comb_idx(virt_space, tup_virt)
    return order_start, occ_start, virt_start


def cluster_tuples_with_nocc(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    cluster_idx: int,
    order: int,
    tup_nocc: int,
) -> Generator[Tuple[np.ndarray, Optional[List[np.ndarray]]], None, None]:
    """
    this function is the main generator for tuples for a given occupation that include
    a certain orbital
    """
    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # orbital
        orb = orb_space[cluster_idx]

        # orbital is occupied
        if orb < nocc:
            # remove orbital
            occ_space = np.delete(occ_space, np.where(occ_space == orb)[0][0])

            # only virtual orbitals
            if tup_nocc == 1:
                for tup_virt in (
                    list(tup) for tup in combinations(virt_space, order - 1)
                ):
                    yield np.array([orb] + tup_virt, dtype=np.int64), None

            # combinations of occupied and virtual MOs
            elif 1 < tup_nocc < order:
                for tup_occ in (
                    list(tup) for tup in combinations(occ_space, tup_nocc - 1)
                ):
                    insort(tup_occ, orb)
                    for tup_virt in (
                        list(tup) for tup in combinations(virt_space, order - tup_nocc)
                    ):
                        yield np.array(tup_occ + tup_virt, dtype=np.int64), None

            # only occupied MOs
            elif tup_nocc == order:
                for tup_occ in (
                    list(tup) for tup in combinations(occ_space, order - 1)
                ):
                    insort(tup_occ, orb)
                    yield np.array(tup_occ, dtype=np.int64), None

        # orbital is virtual
        elif nocc <= orb:
            # remove orbital
            virt_space = np.delete(virt_space, np.where(virt_space == orb)[0][0])

            # only virtual MOs
            if tup_nocc == 0:
                for tup_virt in (
                    list(tup) for tup in combinations(virt_space, order - 1)
                ):
                    insort(tup_virt, orb)
                    yield np.array(tup_virt, dtype=np.int64), None

            # combinations of occupied and virtual MOs
            elif 0 < tup_nocc < order - 1:
                for tup_occ in (list(tup) for tup in combinations(occ_space, tup_nocc)):
                    for tup_virt in (
                        list(tup)
                        for tup in combinations(virt_space, order - 1 - tup_nocc)
                    ):
                        insort(tup_virt, orb)
                        yield np.array(tup_occ + tup_virt, dtype=np.int64), None

            # only occupied MOs
            elif tup_nocc == order - 1:
                for tup_occ in (
                    list(tup) for tup in combinations(occ_space, order - 1)
                ):
                    yield np.array(tup_occ + [orb], dtype=np.int64), None

    else:
        # get cluster
        cluster = orb_clusters[cluster_idx]

        # get cluster size and number of virtual orbitals in cluster
        size = cluster.size
        nvirt = size - cluster.searchsorted(nocc).item()

        # remove cluster from local expansion space
        orb_clusters = orb_clusters[:cluster_idx] + orb_clusters[cluster_idx + 1 :]

        # get number of clusters for every cluster size and number of virtual orbitals
        cluster_info = _cluster_info(orb_clusters, nocc)

        # account for single cluster in tuple
        if size == order and size - nvirt == tup_nocc:
            yield cluster, [cluster]
        # loop over tuples
        else:
            for tup in _recursive_tuples(
                cluster_info,
                orb_clusters,
                order - size,
                order - tup_nocc - nvirt,
                0,
                0,
                0,
            ):
                tup.append(cluster)
                tup_arr = np.concatenate(tup)
                tup_arr.sort()
                yield tup_arr, tup


def tuples_with_nocc(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    order: int,
    tup_nocc: int,
    cached: bool = False,
) -> Generator[np.ndarray, None, None]:
    """
    this function is a generator for tuples for a given number of occupied orbitals,
    setting the cached argument to True accelerates the generator for clusters
    """
    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual orbital spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # only virtual MOs
        if tup_nocc == 0:
            for tup_virt in combinations(virt_space, order):
                yield np.array(tup_virt, dtype=np.int64)
        # combinations of occupied and virtual MOs
        elif 0 < tup_nocc < order:
            for tup_occ in combinations(occ_space, tup_nocc):
                for tup_virt in combinations(virt_space, order - tup_nocc):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64)
        # only occupied MOs
        elif tup_nocc == order:
            for tup_occ in combinations(occ_space, order):
                yield np.array(tup_occ, dtype=np.int64)

    # check if information is supposed to be cached
    elif not cached:
        # get number of clusters for every cluster size and number of virtual
        # orbitals
        cluster_info_list = _cluster_info(orb_clusters, nocc)

        # loop over tuples
        for tup_list in _recursive_tuples(
            cluster_info_list, orb_clusters, order, order - tup_nocc, 0, 0, 0
        ):
            tup = np.concatenate(tup_list)
            tup.sort()
            yield tup

    else:
        # get cluster sizes and number of virtual orbitals
        cluster_sizes = tuple(cluster.size for cluster in orb_clusters)
        cluster_nvirt = tuple(
            cluster.size - cluster.searchsorted(nocc) for cluster in orb_clusters
        )

        # get number of clusters for every cluster size and number of virtual
        # orbitals and corresponding cluster ranges
        cluster_info_tup, cluster_ranges = _cluster_info_and_idx(
            cluster_sizes, cluster_nvirt
        )

        # loop over combinations of different cluster types
        for combination in _recursive_cluster_combinations(
            cluster_info_tup, order, order - tup_nocc, 0, 0, 0
        ):
            # loop over tuples
            for tup_tuple in product(
                *(
                    (
                        np.concatenate(clusters)
                        for clusters in combinations(
                            orb_clusters[
                                cluster_ranges[idx][0] : cluster_ranges[idx][1]
                            ],
                            ncluster,
                        )
                    )
                    for idx, ncluster in combination
                )
            ):
                tup = np.concatenate(tup_tuple)
                tup.sort()
                yield tup


def tuples_idx_with_nocc(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    space_idx: np.ndarray,
    clusters_idx: Optional[List[np.ndarray]],
    nocc: int,
    order: int,
    tup_nocc: int,
    cached: bool = False,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    this function is a generator for tuples and their indices in a given active space
    for a given number of occupied orbitals, setting the cached argument to True
    accelerates the generator for clusters
    """
    # check if only single orbitals in expansion space
    if orb_clusters is None or clusters_idx is None:
        # occupied and virtual orbital spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # occupied and virtual index spaces
        occ_idx = space_idx[: occ_space.size]
        virt_idx = space_idx[occ_space.size :]

        # only virtual MOs
        if tup_nocc == 0:
            for tup_virt, tup_virt_idx in zip(
                combinations(virt_space, order), combinations(virt_idx, order)
            ):
                yield np.array(tup_virt, dtype=np.int64), np.array(
                    tup_virt_idx, dtype=np.int64
                )
        # combinations of occupied and virtual MOs
        elif 0 < tup_nocc < order:
            for tup_occ, tup_occ_idx in zip(
                combinations(occ_space, tup_nocc), combinations(occ_idx, tup_nocc)
            ):
                for tup_virt, tup_virt_idx in zip(
                    combinations(virt_space, order - tup_nocc),
                    combinations(virt_idx, order - tup_nocc),
                ):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64), np.array(
                        tup_occ_idx + tup_virt_idx, dtype=np.int64
                    )
        # only occupied MOs
        elif tup_nocc == order:
            for tup_occ, tup_occ_idx in zip(
                combinations(occ_space, tup_nocc), combinations(occ_idx, tup_nocc)
            ):
                yield np.array(tup_occ, dtype=np.int64), np.array(
                    tup_occ_idx, dtype=np.int64
                )

    # check if information is supposed to be cached
    elif not cached:
        # get number of clusters for every cluster size and number of virtual orbitals
        cluster_info = _cluster_info(orb_clusters, nocc)

        # loop over tuples
        for tup_clusters, tup_clusters_idx in zip(
            _recursive_tuples(
                cluster_info, orb_clusters, order, order - tup_nocc, 0, 0, 0
            ),
            _recursive_tuples(
                cluster_info, clusters_idx, order, order - tup_nocc, 0, 0, 0
            ),
        ):
            tup = np.concatenate(tup_clusters)
            sort_idx = np.argsort(tup)
            tup = tup[sort_idx]
            tup_idx = np.concatenate(tup_clusters_idx)[sort_idx]
            yield tup, tup_idx

    else:
        # get cluster sizes and number of virtual orbitals
        cluster_sizes = tuple(cluster.size for cluster in orb_clusters)
        cluster_nvirt = tuple(
            cluster.size - cluster.searchsorted(nocc) for cluster in orb_clusters
        )

        # get number of clusters for every cluster size and number of virtual
        # orbitals and corresponding cluster ranges
        cluster_info_tup, cluster_ranges = _cluster_info_and_idx(
            cluster_sizes, cluster_nvirt
        )

        # loop over combinations of different cluster types
        for combination in _recursive_cluster_combinations(
            cluster_info_tup, order, order - tup_nocc, 0, 0, 0
        ):
            # loop over tuples
            for tup_tuple, tup_idx_tuple in zip(
                product(
                    *(
                        (
                            np.concatenate(clusters)
                            for clusters in combinations(
                                orb_clusters[
                                    cluster_ranges[idx][0] : cluster_ranges[idx][1]
                                ],
                                ncluster,
                            )
                        )
                        for idx, ncluster in combination
                    )
                ),
                product(
                    *(
                        (
                            np.concatenate(clusters)
                            for clusters in combinations(
                                clusters_idx[
                                    cluster_ranges[idx][0] : cluster_ranges[idx][1]
                                ],
                                ncluster,
                            )
                        )
                        for idx, ncluster in combination
                    )
                ),
            ):
                tup = np.concatenate(tup_tuple)
                sort_idx = np.argsort(tup)
                tup = tup[sort_idx]
                tup_idx = np.concatenate(tup_idx_tuple)[sort_idx]
                yield tup, tup_idx


def tuples_idx_virt_idx_with_nocc(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    space_idx: np.ndarray,
    clusters_idx: Optional[List[np.ndarray]],
    nocc: int,
    order: int,
    tup_nocc: int,
    cached: bool = False,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    this function is a generator for tuples, their indices and the virtual indices in a
    given active space for a given number of occupied orbitals, setting the cached
    argument to True accelerates the generator for clusters
    """
    # check if only single orbitals in orbital space
    if orb_clusters is None or clusters_idx is None:
        # occupied and virtual orbital spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # occupied and virtual index spaces
        occ_idx = space_idx[: occ_space.size]
        virt_idx = space_idx[occ_space.size :]

        # only virtual MOs
        if tup_nocc == 0:
            for tup_virt, tup_virt_idx in zip(
                combinations(virt_space, order), combinations(virt_idx, order)
            ):
                yield np.array(tup_virt, dtype=np.int64), np.array(
                    tup_virt_idx, dtype=np.int64
                ), np.array(tup_virt_idx, dtype=np.int64)
        # combinations of occupied and virtual MOs
        elif 0 < tup_nocc < order:
            for tup_occ, tup_occ_idx in zip(
                combinations(occ_space, tup_nocc), combinations(occ_idx, tup_nocc)
            ):
                for tup_virt, tup_virt_idx in zip(
                    combinations(virt_space, order - tup_nocc),
                    combinations(virt_idx, order - tup_nocc),
                ):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64), np.array(
                        tup_occ_idx + tup_virt_idx, dtype=np.int64
                    ), np.array(tup_virt_idx, dtype=np.int64)
        # only occupied MOs
        elif tup_nocc == order:
            for tup_occ, tup_occ_idx in zip(
                combinations(occ_space, tup_nocc), combinations(occ_idx, tup_nocc)
            ):
                yield np.array(tup_occ, dtype=np.int64), np.array(
                    tup_occ_idx, dtype=np.int64
                ), np.array([], dtype=np.int64)

    # check if information is supposed to be cached
    elif not cached:
        # get number of clusters for every cluster size and number of virtual orbitals
        cluster_info = _cluster_info(orb_clusters, nocc)

        # loop over tuples
        for tup_clusters, tup_clusters_idx in zip(
            _recursive_tuples(
                cluster_info, orb_clusters, order, order - tup_nocc, 0, 0, 0
            ),
            _recursive_tuples(
                cluster_info, clusters_idx, order, order - tup_nocc, 0, 0, 0
            ),
        ):
            tup = np.concatenate(tup_clusters)
            sort_idx = np.argsort(tup)
            tup = tup[sort_idx]
            tup_idx = np.concatenate(tup_clusters_idx)[sort_idx]
            yield tup, tup_idx, tup_idx[tup >= nocc]

    else:
        # get cluster sizes and number of virtual orbitals
        cluster_sizes = tuple(cluster.size for cluster in orb_clusters)
        cluster_nvirt = tuple(
            cluster.size - cluster.searchsorted(nocc) for cluster in orb_clusters
        )

        # get number of clusters for every cluster size and number of virtual
        # orbitals and corresponding cluster ranges
        cluster_info_tup, cluster_ranges = _cluster_info_and_idx(
            cluster_sizes, cluster_nvirt
        )

        # loop over combinations of different cluster types
        for combination in _recursive_cluster_combinations(
            cluster_info_tup, order, order - tup_nocc, 0, 0, 0
        ):
            # loop over tuples
            for tup_tuple, tup_idx_tuple in zip(
                product(
                    *(
                        (
                            np.concatenate(clusters)
                            for clusters in combinations(
                                orb_clusters[
                                    cluster_ranges[idx][0] : cluster_ranges[idx][1]
                                ],
                                ncluster,
                            )
                        )
                        for idx, ncluster in combination
                    )
                ),
                product(
                    *(
                        (
                            np.concatenate(clusters)
                            for clusters in combinations(
                                clusters_idx[
                                    cluster_ranges[idx][0] : cluster_ranges[idx][1]
                                ],
                                ncluster,
                            )
                        )
                        for idx, ncluster in combination
                    )
                ),
            ):
                tup = np.concatenate(tup_tuple)
                sort_idx = np.argsort(tup)
                tup = tup[sort_idx]
                tup_idx = np.concatenate(tup_idx_tuple)[sort_idx]
                yield tup, tup_idx, tup_idx[tup >= nocc]


def _recursive_tuples(
    cluster_info: List[Tuple[int, List[Tuple[int, int]]]],
    clusters: List[np.ndarray],
    remain_norb: int,
    remain_nvirt: int,
    size_idx: int,
    nvirt_idx: int,
    idx: int,
) -> Generator[List[np.ndarray], None, None]:
    """
    this function generates tuples through recursion
    """
    # loop over different cluster size blocks
    for size_idx, (cluster_size, cluster_nvirt_blocks) in enumerate(
        cluster_info[size_idx:], start=size_idx
    ):
        # boolean to break out of loop over nvirt blocks
        break_nvirt = False
        # loop over orbital cluster in size block
        for nvirt_idx, (cluster_nvirt, ncluster) in enumerate(
            cluster_nvirt_blocks[nvirt_idx:], start=nvirt_idx
        ):
            # loop over differing numbers of clusters
            for n in range(1, ncluster + 1):
                # check if active space is smaller than mbe order
                if remain_norb > n * cluster_size:
                    # check if number of added virtual orbitals is less than or equal
                    # to remaining number in tuple
                    if remain_nvirt >= n * cluster_nvirt:
                        # add an additional cluster and subtract cluster size and
                        # number of virtual orbitals from remaining values
                        for tup in _recursive_tuples(
                            cluster_info,
                            clusters,
                            remain_norb - n * cluster_size,
                            remain_nvirt - n * cluster_nvirt,
                            size_idx,
                            nvirt_idx + 1,
                            idx + ncluster,
                        ):
                            # loop over combinations of clusters
                            for cluster_comb in combinations(
                                clusters[idx : idx + ncluster], n
                            ):
                                yield list(cluster_comb) + tup
                    # only single cluster was included
                    elif n == 1:
                        # go to next size block
                        break_nvirt = True
                        break
                    else:
                        # go to next nvirt block
                        break

                # check if active space is equal to mbe order
                elif remain_norb == n * cluster_size:
                    # check if number of added virtual orbitals is equal to remaining
                    # number in tuple
                    if remain_nvirt == n * cluster_nvirt:
                        # loop over combinations of clusters
                        for cluster_comb in combinations(
                            clusters[idx : idx + ncluster], n
                        ):
                            yield list(cluster_comb)
                        # only single cluster was included
                        if n == 1:
                            # go to previous recursion function
                            return
                        # stop adding more clusters
                        break
                    # check if number of added virtual orbitals is larger than remaining
                    # number in tuple
                    elif remain_nvirt < n * cluster_nvirt:
                        # only single cluster was included
                        if n == 1:
                            # go to previous recursion function
                            return
                        # stop adding more clusters
                        break

                # active space is larger than mbe order and only single cluster was
                # included
                elif n == 1:
                    # go to previous recursion function
                    return
                else:
                    # go to next nvirt block
                    break

            # check if breaking out of nvirt block loop
            if break_nvirt:
                # increment cluster index for remaining nvirt blocks
                for _, ncluster in cluster_nvirt_blocks[nvirt_idx:]:
                    idx += ncluster
                break

            # increment cluster index when going to next nvirt block
            idx += ncluster

        # reset nvirt block index
        nvirt_idx = 0


@functools.lru_cache(maxsize=None)
def _recursive_cluster_combinations(
    cluster_info: Tuple[Tuple[int, Tuple[Tuple[int, int]]]],
    remain_norb: int,
    remain_nvirt: int,
    size_idx: int,
    nvirt_idx: int,
    idx: int,
) -> List[List[Tuple[int, int]]]:
    """
    this function generates tuples through recursion
    """
    # intitialize list of cluster combinations
    cluster_combinations = []

    # loop over different cluster size blocks
    for size_idx, (cluster_size, cluster_nvirt_blocks) in enumerate(
        cluster_info[size_idx:], start=size_idx
    ):
        # boolean to break out of loop over nvirt blocks
        break_nvirt = False
        # loop over orbital cluster in size block
        for nvirt_idx, (cluster_nvirt, ncluster) in enumerate(
            cluster_nvirt_blocks[nvirt_idx:], start=nvirt_idx
        ):
            # loop over differing numbers of clusters
            for n in range(1, ncluster + 1):
                # check if active space is smaller than mbe order
                if remain_norb > n * cluster_size:
                    # check if number of added virtual orbitals is less than or equal
                    # to remaining number in tuple
                    if remain_nvirt >= n * cluster_nvirt:
                        # add an additional cluster and subtract cluster size and
                        # number of virtual orbitals from remaining values
                        next_cluster_combinations = _recursive_cluster_combinations(
                            cluster_info,
                            remain_norb - n * cluster_size,
                            remain_nvirt - n * cluster_nvirt,
                            size_idx,
                            nvirt_idx + 1,
                            idx + 1,
                        )

                        # loop over combinations of clusters
                        cluster_combinations += [
                            [(idx, n)] + combs for combs in next_cluster_combinations
                        ]

                    # only single cluster was included
                    elif n == 1:
                        # go to next size block
                        break_nvirt = True
                        break
                    else:
                        # go to next nvirt block
                        break

                # check if active space is equal to mbe order
                elif remain_norb == n * cluster_size:
                    # check if number of added virtual orbitals is equal to remaining
                    # number in tuple
                    if remain_nvirt == n * cluster_nvirt:
                        cluster_combinations.append([(idx, n)])
                        # only single cluster was included
                        if n == 1:
                            # go to previous recursion function
                            return cluster_combinations
                        # stop adding more clusters
                        break
                    # check if number of added virtual orbitals is larger than remaining
                    # number in tuple
                    elif remain_nvirt < n * cluster_nvirt:
                        # only single cluster was included
                        if n == 1:
                            # go to previous recursion function
                            return cluster_combinations
                        # stop adding more clusters
                        break

                # active space is larger than mbe order and only single cluster was
                # included
                elif n == 1:
                    # go to previous recursion function
                    return cluster_combinations
                else:
                    # go to next nvirt block
                    break

            # check if breaking out of nvirt block loop
            if break_nvirt:
                # increment cluster index for remaining nvirt blocks
                idx += len(cluster_nvirt_blocks[nvirt_idx:])
                break

            # increment cluster index when going to next nvirt block
            idx += 1

        # reset nvirt block index
        nvirt_idx = 0

    return cluster_combinations


def _cluster_info(
    cluster_list: List[np.ndarray], nocc: int
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """
    this function returns the number of clusters for all cluster sizes and number of
    virtual orbitals
    """
    curr_size = cluster_list[0].size
    curr_nvirt = curr_size - cluster_list[0].searchsorted(nocc).item()
    curr_ncluster = 1
    cluster_info = []
    curr_list: List[Tuple[int, int]] = []
    for cluster in cluster_list[1:]:
        cluster_nvirt = cluster.size - cluster.searchsorted(nocc).item()
        if cluster.size > curr_size:
            curr_list.append((curr_nvirt, curr_ncluster))
            curr_nvirt = cluster_nvirt
            curr_ncluster = 1
            cluster_info.append((curr_size, curr_list))
            curr_size = cluster.size
            curr_list = []
        elif cluster_nvirt > curr_nvirt:
            curr_list.append((curr_nvirt, curr_ncluster))
            curr_nvirt = cluster_nvirt
            curr_ncluster = 1
        else:
            curr_ncluster += 1
    curr_list.append((curr_nvirt, curr_ncluster))
    cluster_info.append((curr_size, curr_list))

    return cluster_info


@functools.lru_cache(maxsize=None)
def _cluster_info_and_idx(
    cluster_sizes: Tuple[int, ...], cluster_nvirt: Tuple[int, ...]
) -> Tuple[Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...], List[Tuple[int, int]]]:
    """
    this function returns the number of clusters for all cluster sizes and number of
    virtual orbitals and the corresponding cluster index ranges
    """
    idx = start_idx = 0
    curr_size = cluster_sizes[0]
    curr_nvirt = cluster_nvirt[0]
    curr_ncluster = 1
    cluster_info = []
    cluster_idx = []
    curr_list: List[Tuple[int, int]] = []
    for idx, (size, nvirt) in enumerate(
        zip(cluster_sizes[1:], cluster_nvirt[1:]), start=1
    ):
        if size > curr_size:
            cluster_idx.append((start_idx, idx))
            start_idx = idx
            curr_list.append((curr_nvirt, curr_ncluster))
            curr_nvirt = nvirt
            curr_ncluster = 1
            cluster_info.append((curr_size, tuple(curr_list)))
            curr_size = size
            curr_list = []
        elif nvirt > curr_nvirt:
            cluster_idx.append((start_idx, idx))
            start_idx = idx
            curr_list.append((curr_nvirt, curr_ncluster))
            curr_nvirt = nvirt
            curr_ncluster = 1
        else:
            curr_ncluster += 1
    cluster_idx.append((start_idx, idx + 1))
    curr_list.append((curr_nvirt, curr_ncluster))
    cluster_info.append((curr_size, tuple(curr_list)))

    return tuple(cluster_info), cluster_idx


def _comb_idx(space: np.ndarray, tup: np.ndarray) -> int:
    """
    this function return the index of a given (ordered) combination returned from
    itertools.combinations
    """
    idx = _idx(space, tup[0], tup.size)
    idx += sum(
        (
            _idx(space[tup[i - 1] < space], tup[i], tup[i:].size)
            for i in range(1, tup.size)
        )
    )
    return idx


def _idx(space: np.ndarray, idx: int, order: int) -> int:
    """
    this function return the start index of element space[idx] in position (order+1)
    from the right in a given combination
    """
    return sum((comb(space[i < space].size, (order - 1)) for i in space[space < idx]))


def n_tuples(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
) -> int:
    """
    this function returns the total number of tuples of a given order
    """
    # init n_tuples
    n = 0

    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # loop over number of occupied orbitals
        for tup_nocc in range(order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                n += comb(occ_space.size, tup_nocc) * comb(
                    virt_space.size, order - tup_nocc
                )

    else:
        # get number of expansion space clusters for every cluster size and number of
        # virtual orbitals
        size_nvirt_blks = get_ncluster_blks(orb_clusters, nocc)

        # loop over number of occupied orbitals
        for tup_nocc in range(order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                # count tuples
                n += _recursive_ntuples(size_nvirt_blks, 0, 1, order, order - tup_nocc)

    return n


def n_tuples_with_nocc(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    tup_nocc: int,
) -> int:
    """
    this function returns the total number of tuples of a given order and a given
    occupation
    """
    # check if tuple is valid for chosen method
    if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
        # check if only single orbitals in expansion space
        if orb_clusters is None:
            # occupied and virtual expansion spaces
            occ_space = orb_space[orb_space < nocc]
            virt_space = orb_space[nocc <= orb_space]

            return comb(occ_space.size, tup_nocc) * comb(
                virt_space.size, order - tup_nocc
            )

        else:
            # get number of expansion space clusters for every cluster size and number
            # of virtual orbitals
            size_nvirt_blks = get_ncluster_blks(orb_clusters, nocc)

            return _recursive_ntuples(size_nvirt_blks, 0, 1, order, order - tup_nocc)

    else:
        return 0


def n_tuples_predictors(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    cluster_idx: int,
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
) -> Generator[Tuple[int, int, int], None, None]:
    """
    this function yields numbers of tuples for certain predictors
    """
    # single-orbital clusters
    if orb_clusters is None:
        # initialize number of tuples
        ntup = 0

        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # orbital is occupied
        if orb_space[cluster_idx] < nocc:
            # loop over occupations
            for tup_nocc in range(1, min(occ_space.size + 1, order + 1)):
                # check if tuple is valid for chosen method
                if valid_tup(
                    ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc
                ):
                    ntup += comb(occ_space.size - 1, tup_nocc - 1) * comb(
                        virt_space.size, order - tup_nocc
                    )
        # orbital is virtual
        elif nocc <= orb_space[cluster_idx]:
            # loop over occupations
            for tup_nocc in range(min(virt_space.size + 1, order)):
                # check if tuple is valid for chosen method
                if valid_tup(
                    ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc
                ):
                    ntup += comb(occ_space.size, tup_nocc) * comb(
                        virt_space.size - 1, order - tup_nocc - 1
                    )

        # smallest missing contribution is equal to order
        ncluster = order

        # only single current-order contribution is missing
        ncontrib = 1

        yield ntup, ncluster, ncontrib

    # multiple clusters
    elif order > orb_clusters[cluster_idx].size:
        # get cluster size and number of virtual orbitals in cluster
        cluster_size = orb_clusters[cluster_idx].size
        cluster_nvirt = (
            cluster_size - orb_clusters[cluster_idx].searchsorted(nocc).item()
        )

        # get number of expansion space clusters for every cluster size and number of
        # virtual orbitals
        size_nvirt_blks = get_ncluster_blks(
            orb_clusters[:cluster_idx] + orb_clusters[(cluster_idx + 1) :], nocc
        )

        # loop over occupations
        for tup_nocc in range(order + 1):
            # check if tuple is valid for chosen method
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                for (
                    ntup,
                    ncluster,
                    nocc_clusters,
                    nvirt_clusters,
                ) in _recursive_ntuples_predictors(
                    size_nvirt_blks,
                    1,
                    order - cluster_size,
                    order - tup_nocc - cluster_nvirt,
                    0,
                    (),
                    (),
                ):
                    # add cluster
                    ncluster += 1

                    # determine number of missing ncluster orbital contributions
                    ncontrib = n_orb_contrib(
                        np.array(nocc_clusters + (cluster_size - cluster_nvirt,)),
                        np.array(nvirt_clusters + (cluster_nvirt,)),
                        ncluster,
                        ref_nelec,
                        ref_nhole,
                        vanish_exc,
                    )

                    yield ntup, ncluster, ncontrib

    # single cluster
    elif order == orb_clusters[cluster_idx].size:
        # only single tuple contributes for this cluster
        ntup = 1

        # get smallest tuple size that contributes
        ncluster = -(vanish_exc // -2) + 1

        # count contributions
        ncontrib = n_tuples(
            orb_clusters[cluster_idx],
            None,
            nocc,
            ref_nelec,
            ref_nhole,
            vanish_exc,
            ncluster,
        )

        yield ntup, ncluster, ncontrib


@functools.lru_cache(maxsize=None)
def _recursive_ntuples(
    size_nvirt_blks: Tuple[Tuple[int, int, int], ...],
    tot_ntup: int,
    ntup: int,
    remain_norb: int,
    remain_nvirt: int,
) -> int:
    """
    this function generates the number of tuples through recursion
    """
    # loop over cluster blocks with different sizes and number of virtual orbitals
    for idx, (size, nvirt, ncluster) in enumerate(size_nvirt_blks):
        # get remaining blocks
        remain_size_nvirt_blks = size_nvirt_blks[idx + 1 :]
        # loop over differing numbers of clusters
        for n in range(1, ncluster + 1):
            # check if active space is smaller than mbe order
            if remain_norb > n * size:
                # add an additional cluster while adding number of possible tuples
                # and subtracting cluster size from remaining number of orbitals
                tot_ntup = _recursive_ntuples(
                    remain_size_nvirt_blks,
                    tot_ntup,
                    ntup * comb(ncluster, n),
                    remain_norb - n * size,
                    remain_nvirt - n * nvirt,
                )

            # check if active space is equal to mbe order
            elif remain_norb == n * size:
                # check if number of added virtual orbitals is equal to remaining
                # number in tuple
                if remain_nvirt == n * nvirt:
                    # multiply with possible combinations and add to total
                    tot_ntup += ntup * comb(ncluster, n)
                    # only single cluster was included
                    if n == 1:
                        # go to next size block
                        return tot_ntup
                    # stop adding more clusters
                    break
                # check if number of added virtual orbitals is larger than remaining
                # number in tuple
                elif remain_nvirt < n * nvirt:
                    # only single cluster was included
                    if n == 1:
                        # go to next size block
                        return tot_ntup
                    # stop adding more clusters
                    break

            # active space is larger than mbe order and only single cluster was
            # included
            elif n == 1:
                # go to previous recursion function
                return tot_ntup

            # active space is larger than mbe order
            else:
                # stop adding more clusters and go to next nvirt block
                break

    # go to previous recursion function
    return tot_ntup


def _recursive_ntuples_predictors(
    size_nvirt_blks: Tuple[Tuple[int, int, int], ...],
    ntup: int,
    remain_norb: int,
    remain_nvirt: int,
    tup_ncluster: int,
    tup_nocc_clusters: Tuple[int, ...],
    tup_nvirt_clusters: Tuple[int, ...],
) -> Generator[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]], None, None]:
    # loop over cluster blocks with different sizes and number of virtual orbitals
    for idx, (size, nvirt, ncluster) in enumerate(size_nvirt_blks):
        # get remaining blocks
        remain_size_nvirt_blks = size_nvirt_blks[idx + 1 :]
        # loop over differing numbers of clusters
        for n in range(1, ncluster + 1):
            # check if active space is smaller than mbe order
            if remain_norb > n * size:
                # add an additional cluster while adding number of possible tuples
                # and subtracting cluster size from remaining number of orbitals
                yield from _recursive_ntuples_predictors(
                    remain_size_nvirt_blks,
                    ntup * comb(ncluster, n),
                    remain_norb - n * size,
                    remain_nvirt - n * nvirt,
                    tup_ncluster + n,
                    tup_nocc_clusters + n * (size - nvirt,),
                    tup_nvirt_clusters + n * (nvirt,),
                )

            # check if active space is equal to mbe order
            elif remain_norb == n * size:
                # check if number of added virtual orbitals is equal to remaining
                # number in tuple
                if remain_nvirt == n * nvirt:
                    # multiply with possible combinations and yield predictor
                    # combination
                    yield (
                        ntup * comb(ncluster, n),
                        tup_ncluster + n,
                        tup_nocc_clusters + n * (size - nvirt,),
                        tup_nvirt_clusters + n * (nvirt,),
                    )
                    # only single cluster was included
                    if n == 1:
                        # go to next size block
                        return
                    # stop adding more clusters
                    break
                # check if number of added virtual orbitals is larger than remaining
                # number in tuple
                elif remain_nvirt < n * nvirt:
                    # only single cluster was included
                    if n == 1:
                        # go to next size block
                        return
                    # stop adding more clusters
                    break

            # active space is larger than mbe order and only single cluster was
            # included
            elif n == 1:
                # go to previous recursion function
                return

            # active space is larger than mbe order
            else:
                # stop adding more clusters and go to next nvirt block
                break

    # go to previous recursion function
    return


def get_ncluster_blks(
    cluster_list: List[np.ndarray], nocc: int
) -> Tuple[Tuple[int, int, int], ...]:
    """
    this function returns number of clusters by size and number of virtual orbitals
    """
    size_nvirt_blks: List[Tuple[int, int, int]] = []
    last_nvirt = 0
    ncluster = 0
    for idx in range(len(cluster_list)):
        nvirt = (cluster_list[idx] >= nocc).sum()
        if idx > 0 and (
            cluster_list[idx].size > cluster_list[idx - 1].size or nvirt > last_nvirt
        ):
            size_nvirt_blks.append((cluster_list[idx - 1].size, last_nvirt, ncluster))
            ncluster = 1
        else:
            ncluster += 1
        last_nvirt = nvirt
    size_nvirt_blks.append((cluster_list[-1].size, last_nvirt, ncluster))

    return tuple(size_nvirt_blks)


def add_inc_stats(
    abs_inc: float,
    tup: np.ndarray,
    tup_clusters: Optional[List[np.ndarray]],
    adaptive_screen: List[List[np.ndarray]],
    nocc: int,
    order: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
) -> List[List[np.ndarray]]:
    """
    this function adds increment statistics for a given set of predictors
    """
    # single-orbital clusters
    if tup_clusters is None or len(tup_clusters) == order:
        # smallest missing contribution is equal to order
        ncluster = order

        # only single current-order contribution is missing
        ncontrib = 1
    # multiple clusters
    elif len(tup_clusters) > 1:
        # smallest missing contribution is equal to number of clusters
        ncluster = len(tup_clusters)

        # count number of occupied and virtual orbitals in each cluster
        nocc_cluster, nvirt_cluster = cluster_occupation(tup_clusters, nocc)

        # count contributions
        ncontrib = n_orb_contrib(
            nocc_cluster, nvirt_cluster, ncluster, ref_nelec, ref_nhole, vanish_exc
        )
    # single cluster
    else:
        # get smallest tuple size that contributes
        ncluster = -(vanish_exc // -2) + 1

        # count contributions
        ncontrib = n_tuples(tup, None, nocc, ref_nelec, ref_nhole, vanish_exc, ncluster)

    # get logarithmic normalized increment magnitude
    log_norm_abs_inc = np.log(abs_inc) - np.log(ncontrib)

    # add values for increment
    adaptive_screen[ncluster - 1][0][tup] += 1
    adaptive_screen[ncluster - 1][1][tup] += log_norm_abs_inc
    adaptive_screen[ncluster - 1][2][tup] += log_norm_abs_inc**2

    return adaptive_screen


def n_orb_contrib(
    nocc_clusters: np.ndarray,
    nvirt_clusters: np.ndarray,
    ncluster: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
) -> int:
    """
    this function counts the number of n-orbital contributions that are produced by any
    combination of n orbitals where every orbital comes from a distinct cluster from a
    set of n clusters
    """
    # initialize number of orbital contributions equal to number of clusters
    ncontrib = 0

    # minimum and maximum number of occupied orbitals so that every cluster contributes
    min_tup_nocc = ncluster - np.count_nonzero(nvirt_clusters)
    max_tup_nocc = np.count_nonzero(nocc_clusters)

    # loop over number of occupied orbitals in contribution
    for tup_nocc in range(min_tup_nocc, max_tup_nocc + 1):
        # check if number of occupied orbitals produces valid correlation energy
        if valid_tup(ref_nelec, ref_nhole, tup_nocc, ncluster - tup_nocc, vanish_exc):
            # get all combinations by assigning occupied and virtual orbitals to
            # clusters
            norb_combs = np.fromiter(
                chain.from_iterable(
                    chain.from_iterable(
                        zip(
                            combinations(nocc_clusters, tup_nocc),
                            reversed(
                                list(combinations(nvirt_clusters, ncluster - tup_nocc))
                            ),
                        )
                    )
                ),
                dtype=np.int64,
                count=comb(ncluster, tup_nocc) * ncluster,
            ).reshape(-1, ncluster)

            # calculate number of contributions for every combination and sum
            ncontrib += norb_combs.prod(axis=1).sum()

    return ncontrib


def cluster_occupation(
    clusters: List[np.ndarray], nocc: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function outputs the number of occupied orbitals and the number of virtual
    orbitals in each cluster
    """
    # initialize arrays
    nocc_cluster = np.empty(len(clusters), dtype=np.int64)
    nvirt_cluster = np.empty(len(clusters), dtype=np.int64)

    # count number of occupied and virtual orbitals in clusters
    for idx, cluster in enumerate(clusters):
        nocc_cluster[idx] = cluster.searchsorted(nocc)
        nvirt_cluster[idx] = cluster.size - nocc_cluster[idx]

    return nocc_cluster, nvirt_cluster


def cas(ref_space: Optional[np.ndarray], tup: np.ndarray) -> np.ndarray:
    """
    this function returns a cas space
    """
    if ref_space is None:
        return tup
    else:
        return np.sort(np.append(ref_space, tup))


def core_cas(
    nocc: int, ref_space: np.ndarray, tup: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns a core and a cas space
    """
    cas_idx = cas(ref_space, tup)
    core_idx = np.setdiff1d(np.arange(nocc), cas_idx, assume_unique=True)
    return core_idx, cas_idx


def _cas_idx_cart(cas_idx: np.ndarray) -> np.ndarray:
    """
    this function returns a cartesian product of (cas_idx, cas_idx)
    """
    return np.array(np.meshgrid(cas_idx, cas_idx)).T.reshape(-1, 2)


def _coor_to_idx(ij: Tuple[int, int]) -> int:
    """
    this function returns the lower triangular index corresponding to (i, j)
    """
    i = ij[0]
    j = ij[1]
    if i >= j:
        return i * (i + 1) // 2 + j
    else:
        return j * (j + 1) // 2 + i


def idx_tril(cas_idx: np.ndarray) -> np.ndarray:
    """
    this function returns lower triangular cas indices
    """
    cas_idx_cart = _cas_idx_cart(cas_idx)
    return np.unique(
        np.fromiter(
            map(_coor_to_idx, cas_idx_cart),
            dtype=cas_idx_cart.dtype,
            count=cas_idx_cart.shape[0],
        )
    )


def symm_eqv_tup(
    cas_idx: np.ndarray,
    symm_orbs: List[Dict[int, Tuple[int, ...]]],
    ref_space: Optional[np.ndarray],
    clusters: Optional[Dict[int, Tuple[int, ...]]],
) -> Optional[Set[Tuple[int, ...]]]:
    """
    this function returns a set of symmetry-equivalent cas_idx that will yield the same
    CASCI property if the supplied cas_idx is lexicographically greatest with respect
    to all symmetry operations, otherwise returns None
    """
    # add reference space if it is not symmetry-invariant
    cas_idx = cas(ref_space, cas_idx)

    # initialize set of symmetry-equivalent tuples
    eqv_set = {tuple(cas_idx)}

    # loop over symmetry operations in point group
    for symm_op in symm_orbs:
        # get permuted cas space by applying symmetry operation
        perm_cas = apply_symm_op(symm_op, cas_idx, ref_space, clusters)

        # skip this symmetry operation
        if perm_cas is None:
            continue

        # sort permuted cas space
        perm_cas.sort()

        # loop over orbs in cas space and permuted cas space
        for orb, perm_orb in zip(cas_idx, perm_cas):
            # check if orb in cas space is smaller than orb in permuted cas space
            if orb < perm_orb:
                # tuple is not unique and not lexicographically smaller
                return None

            # check if orb in cas space is greater than orb in permuted cas space
            elif orb > perm_orb:
                # tuple is lexicographically greater
                break

        eqv_set.add(tuple(perm_cas))

    return eqv_set


def symm_eqv_inc(
    symm_orbs: np.ndarray,
    eqv_tup_set: Set[Tuple[int, ...]],
    ref_space: Optional[np.ndarray],
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    this function returns a list of sets of tuples that yield symmetrically equivalent
    increments and returns the lexicographically greatest increment for every set
    """
    # initialize list of sets of tuples with the same increments
    eqv_inc_tups = []

    # initialize list of lexicographically greatest tuple for every set
    lex_tups = []

    # check if all tuples in set will produce the same increment
    while len(eqv_tup_set) > 0:
        # start with random tuple in set of equivalent tuples
        curr_cas = eqv_tup_set.pop()

        # get tuple orbitals
        if ref_space is None:
            tup_idx = np.array(curr_cas, dtype=np.int64)
        else:
            tup_idx = np.setdiff1d(curr_cas, ref_space, assume_unique=True)

        # add new set of equivalent increments
        eqv_inc_tups.append([tup_idx])

        # add new lexicographically greatest equivalent increment
        lex_tups.append(tup_idx)

        # get permuted cas spaces by applying symmetry operations
        perm_cas_spaces = symm_orbs[:, curr_cas]

        # sort permuted cas spaces
        perm_cas_spaces.sort()

        # loop over symmetry operations
        for perm_cas in perm_cas_spaces:
            # convert to tuple
            tup_perm_cas = tuple(perm_cas)

            # check if tuple has not been removed from set of symmetrically equivalent
            # tuples
            if tup_perm_cas in eqv_tup_set:
                # get tuple orbitals
                if ref_space is None:
                    tup_idx = np.array(perm_cas, dtype=np.int64)
                else:
                    tup_idx = np.setdiff1d(perm_cas, ref_space, assume_unique=True)

                # add permuted tuple to set of tuples with the same increments
                eqv_inc_tups[-1].append(tup_idx)

                # remove permuted tuple from set of symmetrically equivalent tuples
                eqv_tup_set.remove(tup_perm_cas)

                # loop over orbs in lex_tups and perm_cas
                for lex_orb, perm_orb in zip(lex_tups[-1], perm_cas):
                    # check if orb in lex_tups is smaller than orb in perm_cas
                    if lex_orb < perm_orb:
                        # set permuted cas space as lexicographically greatest
                        lex_tups[-1] = tup_idx

                        # perm_cas is lexicographically greater
                        break

                    # check if orb in lex_tups is greater than orb in perm_cas
                    elif lex_orb > perm_orb:
                        # perm_cas is lexicographically smaller
                        break

    return lex_tups, eqv_inc_tups


def symm_eqv_inc_clusters(
    eqv_inc_lex_tup: List[np.ndarray],
    tup_clusters: Optional[List[np.ndarray]],
    cluster_dict: Optional[Dict[int, Tuple[int, ...]]],
    nocc: int,
) -> List[Optional[List[np.ndarray]]]:
    """
    this function returns a list of clusters for every provided tuple from the set of
    symmetry-equivalent increments
    """
    # use original clusters if only a single tuple is required
    if len(eqv_inc_lex_tup) == 1:
        lex_tups_clusters = [tup_clusters]
    else:
        # check if only single orbital clusters are included in the expansion space
        if cluster_dict is None:
            lex_tups_clusters = [None for _ in eqv_inc_lex_tup]
        else:
            lex_tups_clusters = []
            # generate and sort clusters
            for lex_tup in eqv_inc_lex_tup:

                lex_tups_clusters.append(
                    sorted(
                        [
                            np.array(lex_cluster, dtype=np.int64)
                            for lex_cluster in set(cluster_dict[orb] for orb in lex_tup)
                        ],
                        key=lambda cluster: (cluster.size, (cluster >= nocc).sum())
                        + tuple(orb for orb in cluster),
                    )
                )

    return lex_tups_clusters


def is_lex_tup(
    cas_idx: np.ndarray,
    symm_orbs: np.ndarray,
    ref_space: Optional[np.ndarray],
    clusters: Optional[Dict[int, Tuple[int, ...]]],
) -> bool:
    """
    this function returns whether the supplied cas_idx is the lexicographically
    greatest cas_idx
    """
    # get permuted cas spaces by applying symmetry operations
    perm_cas_spaces = symm_orbs[:, cas_idx]

    # sort permuted cas spaces
    perm_cas_spaces.sort()

    # loop over symmetry operations in point group
    for perm_cas in perm_cas_spaces:
        # skip any incomplete permuted cas spaces
        if perm_cas[0] == -1:
            continue

        # remove reference space
        if ref_space is not None:
            perm_cas = np.setdiff1d(perm_cas, ref_space, assume_unique=True)
            # skip this symmetry operation if full reference space is not included
            if perm_cas.size > cas_idx.size - ref_space.size:
                continue

        # ensure only full clusters are included in permuted tuple
        if clusters is not None:
            try:
                full_clusters = np.unique(
                    np.concatenate([*operator.itemgetter(*perm_cas)(clusters)])
                )
            except ValueError:
                full_clusters = np.array(clusters[perm_cas.item()], dtype=np.int64)
            if full_clusters.size > perm_cas.size:
                continue

        # loop over orbs in cas_idx and perm_cas
        for curr_orb, perm_orb in zip(cas_idx, perm_cas):
            # check if orb in cas_idx is smaller than orb in perm_cas
            if curr_orb < perm_orb:
                # perm_cas is lexicographically greater
                return False

            # check if orb in lex_cas is greater than orb in perm_cas
            elif curr_orb > perm_orb:
                # perm_cas is lexicographically smaller
                break

    return True


def get_lex_tup(
    cas_idx: np.ndarray,
    symm_orbs: np.ndarray,
    ref_space: Optional[np.ndarray],
    clusters: Optional[Dict[int, Tuple[int, ...]]],
) -> np.ndarray:
    """
    this function returns the symmetrically equivalent but lexicographically greater
    cas_idx
    """
    # initialize current lexicographically greatest cas space
    lex_cas = cas_idx

    # get permuted cas spaces by applying symmetry operations
    perm_cas_spaces = symm_orbs[:, cas_idx]

    # sort permuted cas spaces
    perm_cas_spaces.sort()

    # loop over symmetry operations in point group
    for perm_cas in perm_cas_spaces:
        # skip any incomplete permuted cas spaces
        if perm_cas[0] == -1:
            continue

        # remove reference space
        if ref_space is not None:
            perm_cas = np.setdiff1d(perm_cas, ref_space, assume_unique=True)
            # skip this symmetry operation if full reference space is not included
            if perm_cas.size > cas_idx.size - ref_space.size:
                continue

        # ensure only full clusters are included in permuted tuple
        if clusters is not None:
            try:
                full_clusters = np.unique(
                    np.concatenate([*operator.itemgetter(*perm_cas)(clusters)])
                )
            except ValueError:
                full_clusters = np.array(clusters[perm_cas.item()], dtype=np.int64)
            if full_clusters.size > perm_cas.size:
                continue

        # loop over orbs in lex_cas and perm_cas
        for lex_orb, perm_orb in zip(lex_cas, perm_cas):
            # check if orb in lex_cas is smaller than orb in perm_cas
            if lex_orb < perm_orb:
                # set permuted cas space as lexicographically greatest
                lex_cas = perm_cas

                # perm_cas is lexicographically greater
                break

            # check if orb in lex_cas is greater than orb in perm_cas
            elif lex_orb > perm_orb:
                # perm_cas is lexicographically smaller
                break

    return lex_cas


def apply_symm_op(
    symm_op: Dict[int, Tuple[int, ...]],
    tup: np.ndarray,
    ref_space: Optional[np.ndarray],
    clusters: Optional[Dict[int, Tuple[int, ...]]],
) -> Optional[np.ndarray]:
    """
    this function applies a symmetry operation to a tuple of orbitals
    """
    # first generate permuted tuple from symmetry operation
    try:
        perm_set: Set[int] = set()
        try:
            perm_set.update(*operator.itemgetter(*tup)(symm_op))
        except TypeError:
            perm_set.update(operator.itemgetter(*tup)(symm_op))
    except KeyError:
        return None

    # ensure only full clusters are included in permuted tuple
    if clusters is not None:
        try:
            perm_set.update(*operator.itemgetter(*perm_set)(clusters))
        except TypeError:
            perm_set.update(operator.itemgetter(*perm_set)(clusters))

    # ensure full reference space is still included in permuted tuple
    if ref_space is not None:
        perm_set.update(ref_space)

    # only return permuted tuple if length is equal to original tuple size
    set_len = len(perm_set)
    if set_len == tup.size:
        # return permuted tuple
        return np.fromiter(perm_set, np.int64, count=set_len)
    else:
        return None


def reduce_symm_eqv_orbs(
    symm_eqv_orbs: List[Dict[int, Tuple[int, ...]]], space: np.ndarray
) -> List[Dict[int, Tuple[int, ...]]]:
    """
    this function only returns symmetry-equivalent orbitals included in the supplied
    space
    """
    return [
        {
            orb: tuple(tup_orb for tup_orb in tup)
            for orb, tup in symm_op.items()
            if orb in space and np.isin(tup, space).all()
        }
        for symm_op in symm_eqv_orbs
    ]


def get_eqv_inc_orbs(
    symm_eqv_orbs: List[Dict[int, Tuple[int, ...]]], nsymm: int, norb: int
) -> np.ndarray:
    """
    this function returns the orbital combinations necessary to produce a
    symmetry-equivalent increment
    """
    eqv_inc_orbs = np.empty((nsymm, norb), dtype=np.int64)
    for op, symm_op in enumerate(symm_eqv_orbs):
        for orb1 in range(norb):
            orb2 = symm_op.get(orb1, None)
            if orb2 is not None and len(orb2) == 1:
                eqv_inc_orbs[op, orb1] = orb2[0]
            else:
                eqv_inc_orbs[op, orb1] = -1
    return eqv_inc_orbs


def get_nelec(occup: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns the number of electrons in a given tuple of orbitals
    """
    occup_tup = occup[tup]
    return np.array(
        [np.count_nonzero(occup_tup > 0.0), np.count_nonzero(occup_tup > 1.0)]
    )


def get_nhole(nelec: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns the number of holes in a given tuple of orbitals
    """
    return tup.size - nelec


def get_nexc(nelec: np.ndarray, nhole: np.ndarray) -> int:
    """
    this function returns the number of possible excitations in a given tuple of
    orbitals
    """
    return np.add.reduce(np.minimum(nelec, nhole))


def valid_tup(
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    tup_nocc: int,
    tup_nvirt: int,
    vanish_exc: int,
) -> bool:
    """
    this function returns true if a tuple kind produces a non-vanishing correlation
    energy by calculating the number of excitations (equivalent to get_nexc but faster)
    and comparing to the number of vanishing excitation for the used model
    """
    return (
        min(ref_nelec[0] + tup_nocc, ref_nhole[0] + tup_nvirt)
        + min(ref_nelec[1] + tup_nocc, ref_nhole[1] + tup_nvirt)
    ) > vanish_exc


def is_file(string: str, order: int) -> bool:
    """
    this function looks to see if a general restart file corresponding to the input
    string exists
    """
    if order is None:
        return os.path.isfile(os.path.join(RST, f"{string}.npy"))
    else:
        return os.path.isfile(os.path.join(RST, f"{string}_{order}.npy"))


def write_file(
    arr: np.ndarray,
    string: str,
    order: Optional[int] = None,
    nocc: Optional[int] = None,
) -> None:
    """
    this function writes a general restart file corresponding to the input string
    """
    if order is None and nocc is None:
        np.save(os.path.join(RST, f"{string}"), arr)
    elif nocc is None:
        np.save(os.path.join(RST, f"{string}_{order}"), arr)
    else:
        np.save(os.path.join(RST, f"{string}_{order}_{nocc}"), arr)


def write_file_mult(
    arrs: Union[List[np.ndarray], Dict[str, np.ndarray]],
    string: str,
    order: Optional[int] = None,
) -> None:
    """
    this function writes a general restart file corresponding to the input string for
    multiple arrays
    """
    if order is None:
        filename = os.path.join(RST, f"{string}")
    else:
        filename = os.path.join(RST, f"{string}_{order}")

    if isinstance(arrs, list):
        np.savez(os.path.join(RST, filename), *arrs, allow_pickle=False)
    else:
        np.savez(os.path.join(RST, filename), allow_pickle=False, **arrs)


def read_file(string: str, order: Optional[int] = None) -> np.ndarray:
    """
    this function reads a general restart file corresponding to the input string
    """
    if order is None:
        return np.load(os.path.join(RST, f"{string}.npy"))
    else:
        return np.load(os.path.join(RST, f"{string}_{order}.npy"))


def natural_keys(txt: str) -> List[Union[int, str]]:
    """
    this function return keys to sort a string in human order (as
    alist.sort(key=natural_keys))
    see: http://nedbatchelder.com/blog/200712/human_sorting.html
    see: https://stackoverflow.com/questions/5967500
    """
    return [_convert(c) for c in re.split(r"(\d+)", txt)]


def _convert(txt: str) -> Union[int, str]:
    """
    this function converts strings with numbers in them
    """
    return int(txt) if txt.isdigit() else txt


def intervals(a: np.ndarray) -> Generator[List[int], None, None]:
    """
    this generator converts sequential numbers into intervals
    """
    for key, group in groupby(enumerate(a), lambda x: x[1] - x[0]):
        group_lst = list(group)
        if len(group_lst) == 1:
            yield [group_lst[0][1]]
        else:
            yield [group_lst[0][1], group_lst[-1][1]]


def ground_state_sym(orbsym: np.ndarray, nelec: np.ndarray, point_group: str) -> int:
    """
    this function determines the symmetry of the hf ground state
    """
    wfnsym = np.array([0])
    for irrep in orbsym[np.amin(nelec) : np.amax(nelec)]:
        wfnsym = symm.addons.direct_prod(wfnsym, irrep, groupname=point_group)
    return wfnsym.item()


def get_vhf(eri: np.ndarray, nocc: int, norb: int) -> np.ndarray:
    """
    this function determines the Hartree-Fock potential from the electron repulsion
    integrals
    """
    eri = ao2mo.restore(1, eri, norb)

    vhf = np.empty((nocc, norb, norb), dtype=np.float64)
    for i in range(nocc):
        vhf[i] = 2.0 * eri[i, i, :, :] - eri[:, i, i, :]

    return vhf


def get_occup(norb: int, nelec: np.ndarray) -> np.ndarray:
    """
    this function generates the Hartree-Fock occupation vector
    """
    occup = np.zeros(norb, dtype=np.int64)
    occup[: np.amin(nelec)] = 2
    occup[np.amin(nelec) : np.amax(nelec)] = 1

    return occup


def e_core_h1e(
    hcore: np.ndarray, vhf: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    this function returns core energy and cas space 1e integrals
    """
    # determine effective core fock potential
    if core_idx.size > 0:
        core_vhf = np.sum(vhf[core_idx], axis=0)
    else:
        core_vhf = 0.0

    # calculate core energy
    e_core = np.trace((hcore + 0.5 * core_vhf)[core_idx[:, None], core_idx]) * 2.0

    # extract cas integrals
    h1e_cas = (hcore + core_vhf)[cas_idx[:, None], cas_idx]

    return e_core, h1e_cas


def cf_prefactor(contrib_order: int, order: int, max_order: int) -> int:
    """
    this function calculates the prefactor for the closed form MBE
    """
    return comb(max_order - contrib_order, order - contrib_order)


def hop_no_singles(
    solver: Union[fci.direct_spin0_symm.FCI, fci.direct_spin1_symm.FCI],
    norb: int,
    nelec: np.ndarray,
    spin: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    this function generates a special function for the hamiltonian operation where
    singles are omitted
    """
    if spin == 0:
        link_index = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[0])
        na = link_index.shape[0]
        t1_addrs = np.array(
            [
                fci.cistring.str2addr(norb, nelec[0], x)
                for x in fci.cistring.tn_strs(norb, nelec[0], 1)
            ]
        )
        h2e_abs = solver.absorb_h1e(h1e, h2e, norb, nelec, 0.5)

        def hop(c):
            hc = solver.contract_2e(h2e_abs, c.reshape(na, na), norb, nelec, link_index)
            hc[t1_addrs, 0] = 0.0
            hc[0, t1_addrs] = 0.0
            return hc.ravel()

    else:
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[1])
        t1_addrs_a = np.array(
            [
                fci.cistring.str2addr(norb, nelec[0], x)
                for x in fci.cistring.tn_strs(norb, nelec[0], 1)
            ]
        )
        t1_addrs_b = np.array(
            [
                fci.cistring.str2addr(norb, nelec[1], x)
                for x in fci.cistring.tn_strs(norb, nelec[1], 1)
            ]
        )
        h2e_abs = solver.absorb_h1e(h1e, h2e, norb, nelec, 0.5)

        def hop(c):
            hc = solver.contract_2e(h2e_abs, c, norb, nelec, (link_indexa, link_indexb))
            if t1_addrs_a.size > 0:
                hc[t1_addrs_a] = 0.0
            if t1_addrs_b.size > 0:
                hc[t1_addrs_b * na] = 0.0
            return hc.ravel()

    return hop


def get_subspace_det_addr(
    cas: np.ndarray,
    cas_nelec: np.ndarray,
    subspace: np.ndarray,
    subspace_nelec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    this function determines the adresses of subspace determinants in a CAS wavefunction
    """
    # get indices of orbitals inside and outside subspace within cas
    inside_idx = np.searchsorted(cas, subspace)
    outside_idx = np.delete(np.arange(cas.size), inside_idx)

    # get indices of occupied orbitals outside subspace (should be the same regardless
    # of spin if subspace includes all singly occupied orbitals)
    occ_outside_idx = outside_idx[outside_idx < cas_nelec[0]]

    # write bitstring for occupied orbitals outside subspace
    outside_str = 0
    for i in occ_outside_idx:
        outside_str = outside_str | (1 << i)

    # make bitstrings for subspace determinants
    strs_a = fci.cistring.make_strings(inside_idx, subspace_nelec[0])
    strs_b = fci.cistring.make_strings(inside_idx, subspace_nelec[1])

    # add occupied orbitals outside subspace
    strs_a |= outside_str
    strs_b |= outside_str

    # get adresses
    subspace_addr_a = fci.cistring.strs2addr(cas.size, cas_nelec[0], strs_a)
    subspace_addr_b = fci.cistring.strs2addr(cas.size, cas_nelec[1], strs_b)

    # get new signs for coefficients
    sign_a = np.ones(len(strs_a), dtype=np.int64)
    sign_b = np.ones(len(strs_b), dtype=np.int64)
    for orb_idx in occ_outside_idx[::-1]:
        str_shift_a = strs_a >> (orb_idx + 1)
        str_shift_b = strs_b >> (orb_idx + 1)
        sign_a *= bit_parity(str_shift_a)
        sign_b *= bit_parity(str_shift_b)

    return subspace_addr_a, subspace_addr_b, sign_a, sign_b


def init_wfn(norb: int, nelec: np.ndarray, length: int) -> np.ndarray:
    """
    this function initializes a wavefunction
    """
    # determine size of ci vector
    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])

    # initialize ci vector
    civec = np.zeros((length, na, nb))

    return civec


def bit_parity(arr: np.ndarray):
    """
    this function calculates the bit parity of a numpy array
    """
    # try numpy function
    try:
        count = np.bitwise_count(arr)
    # otherwise optimal Hamming weight algorithm
    except:
        s55 = 0x55555555
        s33 = 0x33333333
        s0F = 0x0F0F0F0F
        s01 = 0x01010101

        arr -= (arr >> 1) & s55
        arr = (arr & s33) + ((arr >> 2) & s33)
        arr = (arr + (arr >> 4)) & s0F
        count = (arr * s01) >> 24

    # boolean mask for odd elements
    odd_elements = count % 2 != 0

    # correct sign
    sign = np.ones(arr.size, dtype=np.int64)
    sign[odd_elements] = -1

    return sign


def remove_tup_sq_overlaps(
    tup_sq_overlaps: TupSqOverlapType, min_sq_overlap: float
) -> TupSqOverlapType:
    """
    this function removes all squared overlap values that whitin the upper bound for a
    similar minimum squared overlap value
    """
    ovlps, tups = [], []
    for ovlp, tup in zip(tup_sq_overlaps["overlap"], tup_sq_overlaps["tup"]):
        if ovlp < overlap_range(min_sq_overlap):
            ovlps.append(ovlp)
            tups.append(tup)
    return {"overlap": ovlps, "tup": tups}


def overlap_range(min_sq_overlap: float) -> float:
    """
    this function provides the upper bound for a similar minimum squared overlap value
    """
    return min_sq_overlap + (1.0 - min_sq_overlap) * 1e-2


def flatten_list(lst: List[List[T]]) -> List[T]:
    """
    this function flattens a nested list
    """
    return list(chain.from_iterable(lst))


def adaptive_screen_dict(
    adaptive_screen: List[List[np.ndarray]], norb: int
) -> Dict[str, np.ndarray]:
    """
    this function constructs a dictionary of numpy arrays for the adaptive screening
    information
    """
    array_dict = {
        "inc_count": np.empty((len(adaptive_screen), norb), dtype=np.int64),
        "log_inc_sum": np.empty((len(adaptive_screen), norb), dtype=np.float64),
        "log_inc_sum2": np.empty((len(adaptive_screen), norb), dtype=np.float64),
    }
    for ncluster, screen_info_tup in enumerate(adaptive_screen):
        array_dict["inc_count"][ncluster] = screen_info_tup[0]
        array_dict["log_inc_sum"][ncluster] = screen_info_tup[1]
        array_dict["log_inc_sum2"][ncluster] = screen_info_tup[2]

    return array_dict
