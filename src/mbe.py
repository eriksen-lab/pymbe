#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import sys
import itertools
from pyscf import gto
from typing import Tuple, Set, List, Dict, Union, Any

import kernel
import output
import expansion
import driver
import system
import calculation
import parallel
import tools


def main(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation.CalcCls, exp: expansion.ExpCls, \
            rst_read_a: bool = False, rst_read_b: bool = False, \
            tup_start_a: int = 0, tup_start_b: int = 0, \
            rst_write: bool = False) -> Tuple[Any, ...]:
        """
        this function is the mbe main function
        """
        if mpi.global_master:

            # read restart files
            rst_read_a = tools.is_file(exp.order, 'mbe_idx_a')
            rst_read_b = tools.is_file(exp.order, 'mbe_idx_b')

            # start indices
            tup_start_a = np.asscalar(tools.read_file(exp.order, 'mbe_idx_a')) if rst_read_a else 0
            tup_start_b = np.asscalar(tools.read_file(exp.order, 'mbe_idx_b')) if rst_read_b else 0

            # wake up slaves
            msg = {'task': 'mbe', 'order': exp.order, \
                   'rst_read_a': rst_read_a, 'rst_read_b': rst_read_b, \
                   'tup_start_a': tup_start_a, 'tup_start_b': tup_start_b}
            mpi.global_comm.bcast(msg, root=0)

        # increment dimensions
        dim = tools.inc_dim(calc.target_mbe)

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load hashes for previous orders
        hashes = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target_mbe]['hashes'][k].Shared_query(0)[0] # type: ignore
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k],)))

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=tools.inc_shape(exp.n_tuples['inc'][k], dim)))

        # init time
        if mpi.global_master:
            if not rst_read_a and not rst_read_b:
                exp.time['mbe'].append(0.)
            time = MPI.Wtime()

        # init determinant statistics
        min_ndets = exp.min_ndets[-1] if mpi.global_master and rst_read_a else np.array([1e12], dtype=np.int64)
        max_ndets = exp.max_ndets[-1] if mpi.global_master and rst_read_a else np.array([0], dtype=np.int64)
        sum_ndets = exp.mean_ndets[-1] if mpi.global_master and rst_read_a else np.array([0], dtype=np.int64)

        # init increment statistics
        min_inc = exp.min_inc[-1] if mpi.global_master and rst_read_a else np.array([1.e12] * dim, dtype=np.float64)
        max_inc = exp.max_inc[-1] if mpi.global_master and rst_read_a else np.array([0.] * dim, dtype=np.float64)
        sum_inc = exp.mean_inc[-1] if mpi.global_master and rst_read_a else np.array([0.] * dim, dtype=np.float64)

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < mol.nocc]
        exp_virt = exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]]

        # allow for tuples with only virtual or occupied MOs
        ref_occ = tools.occ_prune(calc.occup, calc.ref_space)
        ref_virt = tools.virt_prune(calc.occup, calc.ref_space)

        # init screen array
        screen = exp.screen if mpi.global_master and rst_read_a else np.ones(mol.norb, dtype=bool)

        # set rst_write
        rst_write = calc.misc['rst'] and mpi.global_size < calc.misc['rst_freq'] < exp.n_tuples['prop'][-1]

        # loop until no tuples left
        for tup_idx, tup in enumerate(itertools.islice(tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order), \
                                        tup_start_a, None), tup_start_a):

            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                continue

            # write restart files and re-init time
            if rst_write and tup_idx % calc.misc['rst_freq'] < mpi.global_size:

                # reduce increment statistics onto global master
                min_inc = parallel.reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
                max_inc = parallel.reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
                sum_inc = parallel.reduce(mpi.global_comm, sum_inc, root=0, op=MPI.SUM)
                if not mpi.global_master:
                    min_inc = np.array([1.e12] * dim, dtype=np.float64)
                    max_inc = np.array([0.] * dim, dtype=np.float64)
                    sum_inc = np.array([0.] * dim, dtype=np.float64)

                # reduce determinant statistics onto global master
                min_ndets = parallel.reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
                max_ndets = parallel.reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
                sum_ndets = parallel.reduce(mpi.global_comm, sum_ndets, root=0, op=MPI.SUM)
                if not mpi.global_master:
                    min_ndets = np.array([1e12], dtype=np.int64)
                    max_ndets = np.array([0], dtype=np.int64)
                    sum_ndets = np.array([0], dtype=np.int64)

                # reduce screen onto global master
                screen = parallel.reduce(mpi.global_comm, screen, root=0, op=MPI.LAND)
                if not mpi.global_master:
                    screen = np.ones(mol.norb, dtype=bool)

                # reduce mbe_idx_a onto global master
                mbe_idx_a = mpi.global_comm.allreduce(tup_idx, op=MPI.MIN)
                # update rst_write
                rst_write = mbe_idx_a + calc.misc['rst_freq'] < exp.n_tuples['prop'][-1] - mpi.global_size

                if mpi.global_master:
                    # write restart files
                    tools.write_file(exp.order, max_inc, 'mbe_max_inc')
                    tools.write_file(exp.order, min_inc, 'mbe_min_inc')
                    tools.write_file(exp.order, sum_inc, 'mbe_mean_inc')
                    tools.write_file(exp.order, max_ndets, 'mbe_max_ndets')
                    tools.write_file(exp.order, min_ndets, 'mbe_min_ndets')
                    tools.write_file(exp.order, sum_ndets, 'mbe_mean_ndets')
                    tools.write_file(exp.order, screen, 'mbe_screen')
                    tools.write_file(exp.order, np.asarray(mbe_idx_a), 'mbe_idx_a')
                    exp.time['mbe'][-1] += MPI.Wtime() - time
                    tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    print(output.mbe_status(mbe_idx_a / exp.n_tuples['prop'][-1]))

            # pi-pruning
            if calc.extra['pi_prune']:
                if not tools.pi_prune(exp.pi_orbs, exp.pi_hashes, tup):
                    continue

            # get core and cas indices
            core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

            # get h2e indices
            cas_idx_tril = tools.cas_idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, ndets_tup, nelec_tup = _inc(calc.model, calc.base['method'], mol.spin, \
                                                 calc.occup, calc.target_mbe, calc.state, calc.orbsym, \
                                                 calc.prop, e_core, h1e_cas, h2e_cas, \
                                                 core_idx, cas_idx, mol.debug, mol.dipole_ints)

            # calculate increment
            if exp.order > exp.min_order:
                inc_tup -= _sum(mol.nocc, calc.target_mbe, exp.min_order, exp.order, \
                                inc, hashes, exp.exp_space, ref_occ, ref_virt, tup)

            # screening procedure
            if calc.target_mbe in ['energy', 'excitation']:
                screen[tup] &= np.abs(inc_tup) < calc.thres['inc']
            else:
                screen[tup] &= np.all(np.abs(inc_tup) < calc.thres['inc'])

            # debug print
            if mol.debug >= 1:
                print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                        ndets_tup, nelec_tup, inc_tup, exp.order, cas_idx, tup))

            # update increment statistics
            min_inc, max_inc, sum_inc = _update(min_inc, max_inc, sum_inc, inc_tup)
            # update determinant statistics
            min_ndets, max_ndets, sum_ndets = _update(min_ndets, max_ndets, sum_ndets, ndets_tup)

        # mpi barrier
        mpi.global_comm.Barrier()

        # increment statistics
        min_inc = parallel.reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
        max_inc = parallel.reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
        sum_inc = parallel.reduce(mpi.global_comm, sum_inc, root=0, op=MPI.SUM)

        # determinant statistics
        min_ndets = parallel.reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
        max_ndets = parallel.reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
        sum_ndets = parallel.reduce(mpi.global_comm, sum_ndets, root=0, op=MPI.SUM)

        # mean increment
        if mpi.global_master:
            mean_inc = sum_inc / exp.n_tuples['prop'][-1]

        # mean number of determinants
        if mpi.global_master:
            mean_ndets = np.asarray(np.rint(sum_ndets / exp.n_tuples['prop'][-1]), dtype=np.int64)

        # print final status
        if mpi.global_master:
            print(output.mbe_status(1.))
            print(output.DIVIDER)

        # collect results on global master
        if mpi.global_master:

            # write restart files
            if calc.misc['rst']:
                tools.write_file(exp.order, max_inc, 'mbe_max_inc')
                tools.write_file(exp.order, min_inc, 'mbe_min_inc')
                tools.write_file(exp.order, sum_inc, 'mbe_mean_inc')
                tools.write_file(exp.order, max_ndets, 'mbe_max_ndets')
                tools.write_file(exp.order, min_ndets, 'mbe_min_ndets')
                tools.write_file(exp.order, sum_ndets, 'mbe_mean_ndets')
                tools.write_file(exp.order, screen, 'mbe_screen')
                tools.write_file(exp.order, np.asarray(exp.n_tuples['prop'][-1]), 'mbe_idx_a')

            # total property
            tot = sum_inc

        # allreduce screened orbitals
        tot_screen = parallel.allreduce(mpi.global_comm, screen, op=MPI.LAND)

        # screen_orbs
        screen_orbs = np.array([mo for mo in np.arange(mol.norb)[tot_screen] if mo in exp.exp_space[-1]], dtype=np.int64)

        # update expansion space wrt screened orbitals
        exp.exp_space.append(np.copy(exp.exp_space[-1]))
        for mo in screen_orbs:
            exp.exp_space[-1] = exp.exp_space[-1][exp.exp_space[-1] != mo]

        # compute updated n_tuples
        n_tuples = tools.n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                  exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                  tools.occ_prune(calc.occup, calc.ref_space), \
                                  tools.virt_prune(calc.occup, calc.ref_space), exp.order)
        # write restart files
        if mpi.global_master:
            tools.write_file(exp.order, np.asarray(n_tuples), 'mbe_n_tuples_inc')

        # init hashes for present order
        if rst_read_b:
            hashes_win = exp.prop[calc.target_mbe]['hashes'][-1]
        else:
            hashes_win = MPI.Win.Allocate_shared(8 * n_tuples if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = hashes_win.Shared_query(0)[0] # type: ignore
        hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tuples,)))
        if mpi.local_master and not mpi.global_master:
            hashes[-1][:].fill(0)

        # init increments for present order
        if rst_read_b:
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
        else:
            inc_win = MPI.Win.Allocate_shared(8 * n_tuples * dim if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = inc_win.Shared_query(0)[0] # type: ignore
        inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=tools.inc_shape(n_tuples, dim)))
        if mpi.local_master and not mpi.global_master:
            inc[-1][:].fill(0.)

        # update occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < mol.nocc]
        exp_virt = exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]]

        # set rst_write
        rst_write = calc.misc['rst'] and mpi.global_size < calc.misc['rst_freq'] < n_tuples

        # mpi barrier
        mpi.local_comm.Barrier()

        # loop until no tuples left
        for tup_idx, tup in enumerate(itertools.islice(tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order), \
                                        tup_start_b, None), tup_start_b):

            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                continue

            # write restart files and re-init time
            if rst_write and (tup_idx % calc.misc['rst_freq']) < mpi.global_size:

                # mpi barrier
                mpi.local_comm.Barrier()

                # reduce hashes & increments onto global master
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[-1][:] = parallel.reduce(mpi.master_comm, hashes[-1], root=0, op=MPI.SUM)
                    if not mpi.global_master:
                        hashes[-1][:].fill(0)
                    inc[-1][:] = parallel.reduce(mpi.master_comm, inc[-1], root=0, op=MPI.SUM)
                    if not mpi.global_master:
                        inc[-1][:].fill(0.)

                # reduce mbe_idx_b onto global master
                mbe_idx_b = mpi.global_comm.allreduce(tup_idx, op=MPI.MIN)
                # update rst_write
                rst_write = mbe_idx_b + calc.misc['rst_freq'] < n_tuples - mpi.global_size

                if mpi.global_master:
                    # write restart files
                    tools.write_file(exp.order, hashes[-1], 'mbe_hashes')
                    tools.write_file(exp.order, inc[-1], 'mbe_inc')
                    tools.write_file(exp.order, np.asarray(mbe_idx_b), 'mbe_idx_b')
                    exp.time['mbe'][-1] += MPI.Wtime() - time
                    tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    print(output.mbe_status(mbe_idx_b / n_tuples))

            # pi-pruning
            if calc.extra['pi_prune']:
                if not tools.pi_prune(exp.pi_orbs, exp.pi_hashes, tup):
                    continue

            # get core and cas indices
            core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

            # get h2e indices
            cas_idx_tril = tools.cas_idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup = _inc(calc.model, calc.base['method'], mol.spin, \
                           calc.occup, calc.target_mbe, calc.state, calc.orbsym, \
                           calc.prop, e_core, h1e_cas, h2e_cas, \
                           core_idx, cas_idx, mol.debug, mol.dipole_ints)[0]

            # calculate increment
            if exp.order > exp.min_order:
                inc_tup -= _sum(mol.nocc, calc.target_mbe, exp.min_order, exp.order, \
                                inc, hashes, exp.exp_space, ref_occ, ref_virt, tup)

            # add hash and increment
            hashes[-1][tup_idx] = tools.hash_1d(tup)
            inc[-1][tup_idx] = inc_tup

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            print(output.mbe_status(1.))

        # allreduce hashes & increments among local masters
        if mpi.local_master:
            hashes[-1][:] = parallel.allreduce(mpi.master_comm, hashes[-1], op=MPI.SUM)
            inc[-1][:] = parallel.allreduce(mpi.master_comm, inc[-1], op=MPI.SUM)

        # sort hashes and increments
        if mpi.local_master:
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1][:].sort()

        # mpi barrier
        mpi.local_comm.Barrier()

        # collect results on global master
        if mpi.global_master:

            # write restart files
            if calc.misc['rst']:
                tools.write_file(exp.order, hashes[-1], 'mbe_hashes')
                tools.write_file(exp.order, inc[-1], 'mbe_inc')
                tools.write_file(exp.order, np.asarray(n_tuples), 'mbe_idx_b')

            # save timing
            exp.time['mbe'][-1] += MPI.Wtime() - time

            return hashes_win, n_tuples, inc_win, tot, \
                    mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc, screen_orbs

        else:

            return hashes_win, n_tuples, inc_win, screen_orbs


def _inc(model: Dict[str, Any], base: Union[str, None], spin: int, occup: np.ndarray, \
         target_mbe: str, state: Dict[str, Any], orbsym: np.ndarray, prop: Dict[str, Any], \
         e_core: float, h1e_cas: np.ndarray, h2e_cas: np.ndarray, core_idx: np.ndarray, \
         cas_idx: np.ndarray, debug: int, dipole_ints: np.ndarray) -> Tuple[Union[float, np.ndarray], \
                                                                            int, Tuple[int, int]]:
        """
        this function calculates the current-order contribution to the increment associated with a given tuple

        example:
        >>> n = 4
        >>> model = {'method': 'fci', 'solver': 'pyscf_spin0', 'hf_guess': True}
        >>> prop = {'hf': {'energy': 0., 'dipole': None}, 'ref': {'energy': 0.}}
        >>> state = {'wfnsym': 'A', 'root': 0}
        >>> occup = np.array([2.] * (n // 2) + [0.] * (n // 2))
        >>> orbsym = np.zeros(n, dtype=np.int64)
        >>> h1e_cas, h2e_cas = kernel.hubbard_h1e((1, n), False), kernel.hubbard_eri((1, n), 2.)
        >>> core_idx, cas_idx = np.array([]), np.arange(n)
        >>> e, ndets, nelec = _inc(model, None, 0, occup, 'energy', state, orbsym,
        ...                        prop, 0, h1e_cas, h2e_cas, core_idx, cas_idx, 0, None)
        >>> np.isclose(e, -2.875942809005048)
        True
        >>> ndets
        36
        >>> nelec
        (2, 2)
        """
        # nelec
        nelec = tools.nelec(occup, cas_idx)

        # perform main calc
        res_full, ndets = kernel.main(model['method'], model['solver'], spin, occup, target_mbe, state['wfnsym'], orbsym, \
                                      model['hf_guess'], state['root'], prop['hf']['energy'], e_core, h1e_cas, h2e_cas, \
                                      core_idx, cas_idx, nelec, debug, dipole_ints, prop['hf']['dipole'])

        # perform base calc
        if base is not None:
            res_full -= kernel.main(base, '', spin, occup, target_mbe, state['wfnsym'], orbsym, \
                                    model['hf_guess'], state['root'], prop['hf']['energy'], e_core, h1e_cas, h2e_cas, \
                                    core_idx, cas_idx, nelec, debug, dipole_ints, prop['hf']['dipole'])[0]

        return res_full - prop['ref'][target_mbe], ndets, nelec


def _sum(nocc: int, target_mbe: str, min_order: int, order: int, \
            inc: List[np.ndarray], hashes: List[np.ndarray], exp_space: List[np.ndarray], \
            ref_occ: bool, ref_virt: bool, tup: np.ndarray) -> Union[float, np.ndarray]:
        """
        this function performs a recursive summation and returns the final increment associated with a given tuple

        example:
        >>> exp_space = [np.arange(10), np.array([1, 2, 3, 4, 5, 7, 8, 9])]
        >>> nocc = 3
        >>> min_order = 2
        >>> ref_occ = False
        >>> ref_virt = False
        >>> hashes = []
        >>> exp_occ = exp_space[0][exp_space[0] < nocc]
        >>> exp_virt = exp_space[0][nocc <= exp_space[0]]
        >>> hashes.append(tools.hash_2d(np.array([tup for tup in tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, 2)])))
        >>> hashes[0].sort()
        >>> exp_occ = exp_space[1][exp_space[1] < nocc]
        >>> exp_virt = exp_space[1][nocc <= exp_space[1]]
        >>> hashes.append(tools.hash_2d(np.array([tup for tup in tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, 3)])))
        >>> hashes[1].sort()
        >>> inc = []
        >>> np.random.seed(1)
        >>> inc.append(np.random.rand(21))
        >>> np.random.seed(2)
        >>> inc.append(np.random.rand(36))
        >>> tup = np.array([1, 7, 8])
        >>> np.isclose(_sum(nocc, 'energy', min_order, tup.size, inc, hashes, exp_space, ref_occ, ref_virt, tup), 1.2177665733781107)
        True
        >>> tup = np.array([1, 7, 8, 9])
        >>> np.isclose(_sum(nocc, 'excitation', min_order, tup.size, inc, hashes, exp_space, ref_occ, ref_virt, tup), 2.7229882355444195)
        True
        >>> np.random.seed(1)
        >>> inc.append(np.random.rand(21, 3))
        >>> np.random.seed(2)
        >>> inc.append(np.random.rand(36, 3))
        >>> tup = np.array([1, 7, 8])
        >>> np.allclose(_sum(nocc, 'dipole', min_order, tup.size, inc, hashes, exp_space, ref_occ, ref_virt, tup),
        ...                 np.array([1.21776657, 1.21776657, 1.21776657]))
        True
        >>> tup = np.array([1, 7, 8, 9])
        >>> np.allclose(_sum(3, 'trans', min_order, 4, inc, hashes, exp_space, ref_occ, ref_virt, tup),
        ...                 np.array([2.72298824, 2.72298824, 2.72298824]))
        True
        """
        # init res
        if target_mbe in ['energy', 'excitation']:
            res = np.zeros(order - min_order, dtype=np.float64)
        else:
            res = np.zeros([order - min_order, 3], dtype=np.float64)

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < nocc]
        tup_virt = tup[nocc <= tup]

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # loop over subtuples
            for tup_sub in tools.tuples(tup_occ, tup_virt, ref_occ, ref_virt, k):

                # compute index
                idx = tools.hash_lookup(hashes[k-min_order], tools.hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    res[k-min_order] += inc[k-min_order][idx]

        return tools.fsum(res)


def _update(min_prop: Union[float, int], max_prop: Union[float, int], \
            sum_prop: Union[float, int], tup_prop: Union[float, int]) -> Tuple[Union[float, int], ...]:
        """
        this function returns updated statistics
        """
        return np.minimum(min_prop, np.abs(tup_prop)), np.maximum(max_prop, np.abs(tup_prop)), sum_prop + tup_prop


if __name__ == "__main__":
    import doctest
    doctest.testmod()



