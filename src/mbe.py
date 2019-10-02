#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe module containing all functions related to MBEs in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import sys
import itertools
from pyscf import gto
from typing import Tuple, List, Dict, Union

import restart
import kernel
import output
import expansion
import driver
import system
import calculation
import parallel
import tools


# tags
class TAGS:
    ready, task, rst, exit = range(4)


def master(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation.CalcCls, exp: expansion.ExpCls) -> Tuple[MPI.Win, float, \
                                                                        float, float, float, \
                                                                        Union[float, np.ndarray], \
                                                                        Union[float, np.ndarray], \
                                                                        Union[float, np.ndarray]]:
        """
        this function is the mbe master function
        """
        # wake up slaves
        msg = {'task': 'mbe', 'order': exp.order}
        mpi.global_comm.bcast(msg, root=0)

        # restart run
        rst_mbe = len(exp.prop[calc.target_mbe]['inc']) == len(exp.hashes)

        # number of slaves
        n_slaves = mpi.global_size - 1

        # init time
        if len(exp.time['mbe']) < len(exp.hashes):
            exp.time['mbe'].append(0.0)
        time = MPI.Wtime()

        # init increments
        if rst_mbe:

            # load restart increments
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
            buf = inc_win.Shared_query(0)[0]
            if calc.target_mbe in ['energy', 'excitation']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],))
            elif calc.target_mbe in ['dipole', 'trans']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3))

        else:

            # new increments
            if calc.target_mbe in ['energy', 'excitation']:
                inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1], 8, comm=mpi.local_comm)
            elif calc.target_mbe in ['dipole', 'trans']:
                inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1] * 3, 8, comm=mpi.local_comm)
            buf = inc_win.Shared_query(0)[0]
            if calc.target_mbe in ['energy', 'excitation']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],))
            elif calc.target_mbe in ['dipole', 'trans']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3))
            inc[:] = np.zeros_like(inc)

        # init determinant statistics
        if rst_mbe:
            min_ndets = exp.min_ndets[-1]
            max_ndets = exp.max_ndets[-1]
            sum_ndets = exp.mean_ndets[-1]
        else:
            min_ndets = 1e12
            max_ndets = 0
            sum_ndets = 0

        # mpi barrier
        mpi.global_comm.Barrier()

        # start index
        if rst_mbe:
           task_start = restart.read_gen(exp.order, 'mbe_idx')
        else:
           task_start = 0

        # loop until no tasks left
        for tup_idx in range(task_start, exp.n_tasks[-1]):

            # probe for available slaves
            mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

            # receive slave status
            mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

            # send tup_idx to slave
            mpi.global_comm.send(tup_idx, dest=mpi.stat.source, tag=TAGS.task)

            # write restart file
            if calc.misc['rst'] and tup_idx > 0 and tup_idx % calc.misc['rst_freq'] == 0:

                # send rst signal to all slaves
                for slave_idx in range(n_slaves):

                    # get slave
                    mpi.global_comm.recv(None, source=slave_idx+1, tag=TAGS.ready)

                    # send rst signal to slave
                    mpi.global_comm.send(None, dest=slave_idx+1, tag=TAGS.rst)

                # mpi barrier
                mpi.global_comm.Barrier()

                # reduce increments onto global master
                if mpi.num_masters > 1:
                    inc[:] = parallel.reduce(mpi.master_comm, inc, root=0)
                min_ndets = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
                max_ndets = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
                sum_ndets = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)

                # save tup_idx
                restart.write_gen(exp.order, np.asarray(tup_idx), 'mbe_idx')

                # save increments
                restart.write_gen(exp.order, inc, 'mbe_inc')

                # save determinant statistics
                restart.write_gen(exp.order, np.asarray(max_ndets), 'mbe_max_ndets')
                restart.write_gen(exp.order, np.asarray(min_ndets), 'mbe_min_ndets')
                restart.write_gen(exp.order, np.asarray(sum_ndets), 'mbe_mean_ndets')

                # save timing
                exp.time['mbe'][-1] += MPI.Wtime() - time
                restart.write_gen(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')

                # re-init time
                time = MPI.Wtime()

                # print status
                print(output.mbe_status(tup_idx / exp.n_tasks[-1]))

        # done with all tasks
        while n_slaves > 0:

            # probe for available slaves
            mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

            # receive slave status
            mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

            # send exit signal to slave
            mpi.global_comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)

            # remove slave
            n_slaves -= 1

        # print final status
        print(output.mbe_status(1.0))

        # mpi barrier
        mpi.global_comm.Barrier()

        # determinant statistics
        min_ndets = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
        max_ndets = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
        sum_ndets = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)

        # mean number of determinants
        mean_ndets = sum_ndets / exp.n_tasks[-1]

        # allreduce increments among local masters
        if mpi.num_masters > 1:
            inc[:] = parallel.allreduce(mpi.master_comm, inc)

        # mpi barrier
        mpi.local_comm.Barrier()

        # save tup_idx
        restart.write_gen(exp.order, np.asarray(exp.n_tasks[-1]), 'mbe_idx')

        # save increments
        restart.write_gen(exp.order, inc, 'mbe_inc')

        # total property
        tot = tools.fsum(inc)

        # statistics
        if calc.target_mbe in ['energy', 'excitation']:

            # increments
            if inc.any():
                mean_inc = np.mean(inc[np.nonzero(inc)])
                min_inc = np.min(np.abs(inc[np.nonzero(inc)]))
                max_inc = np.max(np.abs(inc[np.nonzero(inc)]))
            else:
                mean_inc = min_inc = max_inc = 0.0

        elif calc.target_mbe in ['dipole', 'trans']:

            # init result arrays
            mean_inc = np.empty(3, dtype=np.float64)
            min_inc = np.empty(3, dtype=np.float64)
            max_inc = np.empty(3, dtype=np.float64)

            # loop over x, y, and z
            for k in range(3):

                # increments
                if inc[:, k].any():
                    mean_inc[k] = np.mean(inc[:, k][np.nonzero(inc[:, k])])
                    min_inc[k] = np.min(np.abs(inc[:, k][np.nonzero(inc[:, k])]))
                    max_inc[k] = np.max(np.abs(inc[:, k][np.nonzero(inc[:, k])]))
                else:
                    mean_inc[k] = min_inc[k] = max_inc[k] = 0.0

        # mpi barrier
        mpi.global_comm.Barrier()

        # save timing
        exp.time['mbe'][-1] += MPI.Wtime() - time

        return inc_win, tot, mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc


def slave(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation.CalcCls, exp: expansion.ExpCls) -> MPI.Win:
        """
        this slave function is the mbe slave function
        """
        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load tuples
        buf = exp.tuples.Shared_query(0)[0]
        tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(exp.n_tasks[-1], exp.order))

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0]
            if calc.target_mbe in ['energy', 'excitation']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k],)))
            elif calc.target_mbe in ['dipole', 'trans']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k], 3)))

        # init increments for present order
        if len(exp.prop[calc.target_mbe]['inc']) == len(exp.hashes):
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
            buf = inc_win.Shared_query(0)[0]
        else:
            if mpi.local_master:
                if calc.target_mbe in ['energy', 'excitation']:
                    inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1], 8, comm=mpi.local_comm)
                elif calc.target_mbe in ['dipole', 'trans']:
                    inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1] * 3, 8, comm=mpi.local_comm)
            else:
                inc_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
            buf = inc_win.Shared_query(0)[0]
        if calc.target_mbe in ['energy', 'excitation']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],)))
        elif calc.target_mbe in ['dipole', 'trans']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3)))
        if mpi.local_master and not mpi.global_master:
            inc[-1][:] = np.zeros_like(inc[-1])

        # load hashes for current and previous orders
        hashes = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.hashes[k].Shared_query(0)[0]
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tasks[k],)))

        # init determinant statistics
        min_ndets = 1e12
        max_ndets = 0
        sum_ndets = 0

        # mpi barrier
        mpi.global_comm.Barrier()

        # send availability to master
        mpi.global_comm.send(None, dest=0, tag=TAGS.ready)

        # receive work from master
        while True:

            # probe for task
            mpi.global_comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

            # do task
            if mpi.stat.tag == TAGS.task:

                # receive tup_idx
                tup_idx = mpi.global_comm.recv(source=0, tag=TAGS.task)

                # recover tup
                tup = tuples[tup_idx]

                # get core and cas indices
                core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

                # get h2e indices
                cas_idx_tril = tools.cas_idx_tril(cas_idx)

                # get h2e_cas
                h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                # compute e_core and h1e_cas
                e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                # get inc_idx
                inc_idx = tools.hash_compare(hashes[-1], tools.hash_1d(tup))

                # calculate increment
                inc_tup, ndets, nelec = _inc(calc.model['method'], calc.base['method'], calc.model['solver'], \
                                               calc.occup, calc.target_mbe, calc.state['wfnsym'], calc.orbsym, \
                                               calc.extra['hf_guess'], calc.state['root'], calc.prop['hf']['energy'], \
                                               calc.prop['ref'][calc.target_mbe], e_core, h1e_cas, h2e_cas, \
                                               core_idx, cas_idx, mol.debug, \
                                               mol.dipole if calc.target_mbe in ['dipole', 'trans'] else None, \
                                               calc.mo_coeff if calc.target_mbe in ['dipole', 'trans'] else None, \
                                               calc.prop['hf']['dipole'] if calc.target_mbe in ['dipole', 'trans'] else None)

                # calculate increment
                if exp.order > exp.min_order:
                    if np.any(inc_tup != 0.0):
                        inc_tup -= _sum(calc.occup, calc.ref_space, calc.exp_space, calc.target_mbe, \
                                        exp.min_order, exp.order, inc, hashes, tup, pi_prune=calc.extra['pi_prune'])


                # debug print
                if mol.debug >= 1:
                    print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                            ndets, nelec, inc_tup, exp.order, cas_idx, tup))

                # add to inc
                inc[-1][inc_idx] = inc_tup

                # update determinant statistics
                min_ndets = min(min_ndets, ndets)
                max_ndets = max(max_ndets, ndets)
                sum_ndets += ndets

                # send availability to master
                mpi.global_comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.rst:

                # receive rst signal
                mpi.global_comm.recv(None, source=0, tag=TAGS.rst)

                # mpi barrier
                mpi.global_comm.Barrier()

                # reduce increments onto global master
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[-1][:] = parallel.reduce(mpi.master_comm, inc[-1], root=0)
                    inc[-1][:] = np.zeros_like(inc[-1])
                _ = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
                _ = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
                _ = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)
                sum_ndets = 0

                # send availability to master
                mpi.global_comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.exit:

                # receive exit signal
                mpi.global_comm.recv(None, source=0, tag=TAGS.exit)

                break

        # mpi barrier
        mpi.global_comm.Barrier()

        # determinant statistics
        min_ndets = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
        max_ndets = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
        sum_ndets = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)

        # allreduce increments among local masters
        if mpi.num_masters > 1 and mpi.local_master:
            inc[-1][:] = parallel.allreduce(mpi.master_comm, inc[-1])

        # mpi barrier
        mpi.local_comm.Barrier()

        # mpi barrier
        mpi.global_comm.Barrier()

        return inc_win


def _inc(main_method: str, base_method: Union[str, None], solver: str, \
            occup: np.ndarray, target_mbe: str, state_wfnsym: str, orbsym: np.ndarray, hf_guess: bool, \
            state_root: int, e_hf: float, e_ref: float, e_core: float, h1e_cas: np.ndarray, \
            h2e_cas: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, \
            debug: int, ao_dipole: Union[np.ndarray, None], mo_coeff: Union[np.ndarray, None], \
            dipole_hf: Union[np.ndarray, None]) -> Tuple[Union[float, np.ndarray], int, Tuple[int, int]]:
        """
        this function calculates the current-order contribution to the increment associated with a given tuple

        example:
        >>> n = 4
        >>> occup = np.array([2.] * (n // 2) + [0.] * (n // 2))
        >>> orbsym = np.zeros(n, dtype=np.int)
        >>> h1e_cas, h2e_cas = kernel.hubbard_h1e((1, n), n, False), kernel.hubbard_eri(n, 2.)
        >>> core_idx, cas_idx = np.array([]), np.arange(n)
        >>> _inc('fci', None, 'pyscf_spin0', occup, 'energy', None, orbsym, 0,
        ...      0, 0., 0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, False, None, None, None)
        (-2.875942809005048, 36, (2, 2))
        """
        # nelec
        nelec = tools.nelec(occup, cas_idx)

        # perform main calc
        e_full, ndets = kernel.main(main_method, solver, occup, target_mbe, state_wfnsym, orbsym, \
                                        hf_guess, state_root, e_hf, e_core, h1e_cas, h2e_cas, \
                                        core_idx, cas_idx, nelec, debug, ao_dipole, mo_coeff, dipole_hf)

        # perform base calc
        if base_method is not None:
            e_full -= kernel.main(base_method, solver, occup, target_mbe, state_wfnsym, orbsym, \
                                      hf_guess, state_root, e_hf, e_core, h1e_cas, h2e_cas, \
                                      core_idx, cas_idx, nelec, debug, ao_dipole, mo_coeff, dipole_hf)[0]

        return e_full - e_ref, ndets, nelec


def _sum(occup: np.ndarray, ref_space: np.ndarray, exp_space: Dict[str, np.ndarray], \
            target_mbe: str, min_order: int, order: int, inc: List[np.ndarray], \
            hashes: List[np.ndarray], tup: np.ndarray, pi_prune: bool = False) -> Union[float, np.ndarray]:
        """
        this function performs a recursive summation and returns the final increment associated with a given tuple

        example:
        >>> occup = np.array([2.] * 2 + [0.] * 2)
        >>> ref_space = np.arange(2)
        >>> exp_space = {'tot': np.arange(2, 4, dtype=np.int16),
        ...              'occ': np.array([], dtype=np.int16),
        ...              'virt': np.arange(2, 4, dtype=np.int16),
        ...              'pi_orbs': None, 'pi_hashes': None}
        >>> min_order, order = 1, 2
        ... # [[2], [3]]
        ... # [[2, 3]]
        >>> hashes = [np.sort(np.array([-4760325697709127167, -4199509873246364550])),
        ...           np.array([-5475322122992870313])]
        >>> inc = [np.array([-.1, -.2])]
        >>> tup = np.arange(2, 4, dtype=np.int32)
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, False), -.3)
        True
        >>> inc = [np.array([[0., 0., .1], [0., .0, .2]])]
        >>> np.allclose(_sum(occup, ref_space, exp_space, 'dipole', min_order, order, inc, hashes, tup, False), np.array([0., 0., .3]))
        True
        >>> ref_space = np.array([])
        >>> exp_space = {'tot': np.arange(4), 'occ': np.arange(2), 'virt': np.arange(2, 4),
        ...              'pi_orbs': np.arange(2, dtype=np.int32), 'pi_hashes': np.array([-3821038970866580488])}
        >>> min_order, order = 2, 4
        ... # [[0, 2], [0, 3], [1, 2], [1, 3]]
        ... # [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        ... # [[0, 1, 2, 3]]
        >>> hashes = [np.sort(np.array([-4882741555304645790, 1455941523185766351, -2163557957507198923, -669804309911520350])),
        ...           np.sort(np.array([-5731810011007442268, 366931854209709639, -7216722148388372205, -3352798558434503475])),
        ...           np.array([-2930228190932741801])]
        >>> inc = [np.array([-.11, -.12, -.11, -.12]), np.array([-.01, -.02, -.03, -.03])]
        >>> tup = np.arange(4, dtype=np.int32)
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, False), -0.55)
        True
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, True), -0.05)
        True
        """
        # init res
        if target_mbe in ['energy', 'excitation']:
            res = 0.0
        else:
            res = np.zeros(3, dtype=np.float64)

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # generate array with all subsets of particular tuple
            combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int16)

            # prune combinations without a mix of occupied and virtual orbitals
            if min_order == 2:
                combs = combs[np.fromiter(map(functools.partial(tools.corr_prune, occup), combs), \
                                              dtype=bool, count=combs.shape[0])]

            # prune combinations with non-degenerate pairs of pi-orbitals
            if pi_prune:
                combs = combs[np.fromiter(map(functools.partial(tools.pi_prune, \
                                              exp_space['pi_orbs'], exp_space['pi_hashes']), combs), \
                                              dtype=bool, count=combs.shape[0])]

            if combs.size == 0:
                continue

            # convert to sorted hashes
            combs_hash = tools.hash_2d(combs)
            combs_hash.sort()

            # get indices of combinations
            idx = tools.hash_compare(hashes[k-min_order], combs_hash)

            # assertion
            tools.assertion(idx is not None, 'error in recursive increment calculation:\n'
                                             'k = {:}\ntup:\n{:}\ncombs:\n{:}'. \
                                             format(k, tup, combs))

            # add up lower-order increments
            res += tools.fsum(inc[k-min_order][idx])

        return res


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)



