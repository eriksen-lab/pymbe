#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
parallel module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

from mpi4py import MPI
import numpy as np
from pyscf import lib
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Tuple, Optional

    from pymbe.pymbe import MBE


# parameters for tiled mpi operations
INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1


class MPICls:
    """
    this class contains the pymbe mpi attributes
    """

    def __init__(self) -> None:
        """
        init mpi attributes
        """
        # general
        self.host = MPI.Get_processor_name()
        self.stat = MPI.Status()

        # global communicator
        self.global_comm = MPI.COMM_WORLD
        self.global_size = self.global_comm.Get_size()
        self.global_rank = self.global_comm.Get_rank()
        self.global_master = self.global_rank == 0

        # local node communicator (memory sharing)
        self.local_comm = cast(
            MPI.Intracomm,
            self.global_comm.Split_type(MPI.COMM_TYPE_SHARED, self.global_rank),
        )
        self.local_rank = self.local_comm.Get_rank()
        self.local_master = self.local_rank == 0

        # master communicator
        self.master_comm = self.global_comm.Split(
            1 if self.local_master else MPI.UNDEFINED, self.global_rank
        )

        # local masters
        if self.local_master:
            self.num_masters = self.master_comm.Get_size()
            self.master_global_ranks = np.array(
                self.master_comm.allgather(self.global_rank)
            )
            self.master_global_hosts = np.array(self.master_comm.allgather(self.host))

        # number of masters
        if self.global_master:
            self.global_comm.bcast(self.num_masters, root=0)
        else:
            self.num_masters = self.global_comm.bcast(None, root=0)


def kw_dist(mbe: MBE) -> MBE:
    """
    this function bcast all keywords to slaves
    """
    if mbe.mpi.global_master:
        # collect keywords (must be updated with new future attributes)
        keywords = [
            "method",
            "target",
            "point_group",
            "fci_state_sym",
            "fci_state_root",
            "orb_type",
            "ref_thres",
            "base_method",
            "screen_type",
            "screen_start",
            "screen_perc",
            "screen_thres",
            "screen_func",
            "max_order",
            "rst",
            "rst_freq",
            "restarted",
            "fci_backend",
            "cc_backend",
            "hf_guess",
            "verbose",
            "dryrun",
            "no_singles",
        ]

        # put keyword attributes that exist into dictionary
        kw_dict = {}
        for kw in keywords:
            if hasattr(mbe, kw):
                kw_dict[kw] = getattr(mbe, kw)

        # bcast to slaves
        mbe.mpi.global_comm.bcast(kw_dict, root=0)

    else:
        # receive info from master
        kw_dict = mbe.mpi.global_comm.bcast(None, root=0)

        # set mbe attributes from keywords
        for key, val in kw_dict.items():
            setattr(mbe, key, val)

    return mbe


def system_dist(mbe: MBE) -> MBE:
    """
    this function bcasts all system quantities to slaves
    """
    if mbe.mpi.global_master:
        # collect system quantites (must be updated with new future attributes)
        system = [
            "norb",
            "nelec",
            "orbsym",
            "ref_space",
            "exp_space",
            "base_prop",
            "dipole_ints",
            "full_norb",
            "full_nocc",
            "inact_fock",
            "eri_goaa",
            "eri_gaao",
            "eri_gaaa",
        ]

        # put keyword attributes that exist into dictionary
        system_dict = {}
        for attr in system:
            if hasattr(mbe, attr):
                system_dict[attr] = getattr(mbe, attr)

        # bcast to slaves
        mbe.mpi.global_comm.bcast(system_dict, root=0)

    else:
        # receive system quantites from master
        system_dict = mbe.mpi.global_comm.bcast(None, root=0)

        # set mbe attributes from system dict
        for key, val in system_dict.items():
            setattr(mbe, key, val)

    return mbe


def mpi_bcast(comm: MPI.Comm, buff: np.ndarray) -> np.ndarray:
    """
    this function performs a tiled Bcast operation
    inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
    """
    # init buff_tile
    buff_tile: np.ndarray = np.ndarray(buff.size, dtype=buff.dtype, buffer=buff)

    # bcast all tiles
    for p0, p1 in lib.prange(0, buff.size, BLKSIZE):
        comm.Bcast(buff_tile[p0:p1], root=0)

    return buff


def mpi_reduce(
    comm: MPI.Comm, send_buff: np.ndarray, root: int = 0, op: MPI.Op = MPI.SUM
) -> np.ndarray:
    """
    this function performs a tiled Reduce operation
    inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
    """
    # rank
    rank = comm.Get_rank()

    # init recv_buff
    if rank == root:
        recv_buff = np.zeros_like(send_buff)
    else:
        recv_buff = send_buff

    # init send_tile and recv_tile
    send_tile: np.ndarray = np.ndarray(
        send_buff.size, dtype=send_buff.dtype, buffer=send_buff
    )
    if rank == root:
        recv_tile: np.ndarray = np.ndarray(
            recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff
        )

    # reduce all tiles
    for p0, p1 in lib.prange(0, send_buff.size, BLKSIZE):
        if rank == root:
            comm.Reduce(send_tile[p0:p1], recv_tile[p0:p1], op=op, root=root)
        else:
            comm.Reduce(send_tile[p0:p1], None, op=op, root=root)

    return recv_buff


def mpi_allreduce(
    comm: MPI.Comm, send_buff: np.ndarray, op: MPI.Op = MPI.SUM
) -> np.ndarray:
    """
    this function performs a tiled Allreduce operation
    inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
    """
    # init recv_buff
    recv_buff = np.zeros_like(send_buff)

    # init send_tile and recv_tile
    send_tile: np.ndarray = np.ndarray(
        send_buff.size, dtype=send_buff.dtype, buffer=send_buff
    )
    recv_tile: np.ndarray = np.ndarray(
        recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff
    )

    # allreduce all tiles
    for p0, p1 in lib.prange(0, send_buff.size, BLKSIZE):
        comm.Allreduce(send_tile[p0:p1], recv_tile[p0:p1], op=op)

    return recv_buff


def open_shared_win(
    window: Optional[MPI.Win], dtype: type, shape: Tuple[int, ...]
) -> np.ndarray:
    """
    this function returns the numpy array to a MPI window
    """
    if window is not None:
        shared = np.ndarray(
            buffer=window.Shared_query(0)[0],  # type: ignore
            dtype=dtype,
            shape=shape,
        )
    else:
        shared = np.empty(shape=shape, dtype=dtype)

    return shared
