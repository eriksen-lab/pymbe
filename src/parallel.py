#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
parallel module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import shutil
try:
    from mpi4py import MPI
except ImportError:
    sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
    import numpy as np
except ImportError:
    sys.stderr.write('\nImportError : numpy module not found\n\n')
from pyscf import symm, lib
from typing import Tuple

from system import MolCls
from calculation import CalcCls


# restart folder
RST = os.getcwd()+'/rst'
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
                self.global_master = (self.global_rank == 0)

                # local node communicator (memory sharing)
                self.local_comm = self.global_comm.Split_type(MPI.COMM_TYPE_SHARED, self.global_rank)
                self.local_rank = self.local_comm.Get_rank()
                self.local_master = (self.local_rank == 0)

                # master communicator
                self.master_comm = self.global_comm.Split(1 if self.local_master else MPI.UNDEFINED, self.global_rank)

                # local masters
                if self.local_master:

                    self.num_masters = self.master_comm.Get_size()
                    self.master_global_ranks = np.array(self.master_comm.allgather(self.global_rank))
                    self.master_global_hosts = np.array(self.master_comm.allgather(self.host))

                # number of masters
                if self.global_master:
                    self.global_comm.bcast(self.num_masters, root=0)
                else:
                    self.num_masters = self.global_comm.bcast(None, root=0)


def mol_dist(mpi: MPICls, mol: MolCls) -> MolCls:
        """
        this function bcast all standard mol info to slaves
        """
        if mpi.global_master:

            # collect standard info (must be updated with new future attributes)
            info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, 'x2c': mol.x2c, \
                    'ncore': mol.ncore, 'symmetry': mol.symmetry, 'basis': mol.basis, 'gauge': mol.gauge, \
                    'cart': mol.cart, 'unit': mol.unit, 'frozen': mol.frozen, 'debug': mol.debug}

            # add hubbard info if relevant (also needs to be updated with new future attributes)
            if not mol.atom:

                info['u'] = mol.u
                info['n'] = mol.n
                info['matrix'] = mol.matrix
                info['pbc'] = mol.pbc
                info['nelectron'] = mol.nelectron

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set mol attributes from info dict
            for key, val in info.items():
                setattr(mol, key, val)

        return mol


def calc_dist(mpi: MPICls, calc: CalcCls) -> CalcCls:
        """
        this function bcast all standard calc info to slaves
        """
        if mpi.global_master:

            # collect standard info (must be updated with new future attributes)
            info = {'model': calc.model, 'target_mbe': calc.target_mbe, 'base': calc.base, \
                    'thres': calc.thres, 'state': calc.state, \
                    'extra': calc.extra, 'misc': calc.misc, 'mpi': calc.mpi, \
                    'orbs': calc.orbs, 'restart': calc.restart}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set calc attributes from info dict
            for key, val in info.items():
                setattr(calc, key, val)

        return calc


def fund_dist(mpi: MPICls, mol: MolCls, calc: CalcCls) -> Tuple[MolCls, CalcCls]:
        """
        this function bcasts all fundamental mol and calc info to slaves
        """
        if mpi.global_master:

            # collect standard info (must be updated with new future attributes)
            info = {'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, 'e_nuc': mol.e_nuc}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

            # collect standard info (must be updated with new future attributes)
            info = {'occup': calc.occup, 'ref_space': calc.ref_space}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

            # bcast mo coefficients
            calc.mo_coeff = mpi_bcast(mpi.global_comm, calc.mo_coeff)

            # update orbsym
            if mol.symmetry:
                calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
            else:
                calc.orbsym = np.zeros(mol.norb, dtype=np.int64)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set mol attributes from info dict
            for key, val in info.items():
                setattr(mol, key, val)

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set calc attributes from info dict
            for key, val in info.items():
                setattr(calc, key, val)

            # receive mo coefficients
            calc.mo_coeff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
            calc.mo_coeff = mpi_bcast(mpi.global_comm, calc.mo_coeff)

            # update orbsym
            if mol.symmetry:
                calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
            else:
                calc.orbsym = np.zeros(mol.norb, dtype=np.int64)

        return mol, calc


def prop_dist(mpi: MPICls, calc: CalcCls) -> CalcCls:
        """
        this function bcasts properties to slaves
        """
        if mpi.global_master:

            # bcast to slaves
            mpi.global_comm.bcast(calc.prop, root=0)

        else:

            # receive prop from master
            calc.prop = mpi.global_comm.bcast(None, root=0)

        return calc


def mpi_bcast(comm: MPI.Comm, buff: np.ndarray) -> np.ndarray:
        """
        this function performs a tiled Bcast operation
        inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
        """
        # init buff_tile
        buff_tile = np.ndarray(buff.size, dtype=buff.dtype, buffer=buff)

        # bcast all tiles
        for p0, p1 in lib.prange(0, buff.size, BLKSIZE):
            comm.Bcast(buff_tile[p0:p1], root=0)

        return buff


def mpi_reduce(comm: MPI.Comm, send_buff: np.ndarray, root: int = 0, op: MPI.Op = MPI.SUM) -> np.ndarray:
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
        send_tile = np.ndarray(send_buff.size, dtype=send_buff.dtype, buffer=send_buff)
        if rank == root:
            recv_tile = np.ndarray(recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff)

        # reduce all tiles
        for p0, p1 in lib.prange(0, send_buff.size, BLKSIZE):
            if rank == root:
                comm.Reduce(send_tile[p0:p1], recv_tile[p0:p1], op=op, root=root)
            else:
                comm.Reduce(send_tile[p0:p1], None, op=op, root=root)

        return recv_buff


def mpi_allreduce(comm: MPI.Comm, send_buff: np.ndarray, op: MPI.Op = MPI.SUM) -> np.ndarray:
        """
        this function performs a tiled Allreduce operation
        inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
        """
        # init recv_buff        
        recv_buff = np.zeros_like(send_buff)

        # init send_tile and recv_tile
        send_tile = np.ndarray(send_buff.size, dtype=send_buff.dtype, buffer=send_buff)
        recv_tile = np.ndarray(recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff)

        # allreduce all tiles
        for p0, p1 in lib.prange(0, send_buff.size, BLKSIZE):
            comm.Allreduce(send_tile[p0:p1], recv_tile[p0:p1], op=op)

        return recv_buff


def mpi_gatherv(comm: MPI.Comm, send_buff: np.ndarray, recv_buff: np.ndarray, \
                counts: np.ndarray, root: int = 0) -> np.ndarray:
        """
        this function performs a gatherv operation using point-to-point operations
        inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py
        """
        # rank and size
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == root:
            recv_tile = np.ndarray(recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff)
            recv_tile[:send_buff.size] = send_buff
            # recv from all slaves
            for slave in range(1, size):
                slave_idx = np.sum(counts[:slave])
                comm.Recv(recv_tile[slave_idx:slave_idx+counts[slave]], source=slave)
        else:
            comm.Send(send_buff, dest=root)

        return recv_buff


def mpi_finalize(mpi: MPICls, rst_write: bool) -> None:
        """
        this function terminates a successful pymbe calculation
        """
        # wake up slaves
        if mpi.global_master:
            if rst_write:
                shutil.rmtree(RST)
            mpi.global_comm.bcast({'task': 'exit'}, root=0)

        # finalize
        mpi.global_comm.Barrier()
        MPI.Finalize()


