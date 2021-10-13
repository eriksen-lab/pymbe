#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI

from mbe import main as mbe_main
from output import main_header, mbe_header, mbe_results, mbe_end, \
                    screen_results, purge_header, purge_results, purge_end
from purge import main as purge_main
from system import MolCls
from calculation import CalcCls
from expansion import ExpCls
from parallel import MPICls, mpi_finalize
from tools import n_tuples, occ_prune, virt_prune, inc_dim, inc_shape, write_file
from ml import train


def master(mpi: MPICls, mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
        """
        this function is the main pymbe master function
        """
        # print expansion headers
        print(main_header(mpi=mpi, method=calc.model['method']))

        # print output from restarted calculation
        if calc.restart:
            for i in range(exp.min_order, exp.start_order):

                # print mbe header
                print(mbe_header(i, exp.n_tuples['calc'][i-exp.min_order], \
                                 1. if i < calc.thres['start'] else calc.thres['perc']))

                # print mbe end
                print(mbe_end(i, exp.time['mbe'][i-exp.min_order]))

                # print mbe results
                print(mbe_results(calc.target_mbe, calc.state['root'], exp.min_order, i, \
                                  exp.prop[calc.target_mbe]['tot'], exp.mean_inc[i-exp.min_order], \
                                  exp.min_inc[i-exp.min_order], exp.max_inc[i-exp.min_order], \
                                  exp.mean_ndets[i-exp.min_order], exp.min_ndets[i-exp.min_order], \
                                  exp.max_ndets[i-exp.min_order]))

                # print screening results
                exp.screen_orbs = np.setdiff1d(exp.exp_space[i-exp.min_order], exp.exp_space[i-exp.min_order+1])
                if 0 < exp.screen_orbs.size:
                    print(screen_results(i, exp.screen_orbs, exp.exp_space))

        # begin or resume mbe expansion depending
        for exp.order in range(exp.start_order, exp.max_order+1):

            # theoretical and actual number of tuples at current order
            if len(exp.n_tuples['inc']) == exp.order - exp.min_order:
                exp.n_tuples['theo'].append(n_tuples(exp.exp_space[0][exp.exp_space[0] < mol.nocc], \
                                                     exp.exp_space[0][mol.nocc <= exp.exp_space[0]], \
                                                     occ_prune(calc.occup, calc.ref_space), \
                                                     virt_prune(calc.occup, calc.ref_space), exp.order))
                exp.n_tuples['calc'].append(n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                                    exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                                    occ_prune(calc.occup, calc.ref_space), \
                                                    virt_prune(calc.occup, calc.ref_space), exp.order))
                exp.n_tuples['inc'].append(exp.n_tuples['calc'][-1])
                if calc.misc['rst']:
                    write_file(exp.order, np.asarray(exp.n_tuples['theo'][-1]), 'mbe_n_tuples_theo')
                    write_file(exp.order, np.asarray(exp.n_tuples['calc'][-1]), 'mbe_n_tuples_calc')
                    write_file(exp.order, np.asarray(exp.n_tuples['inc'][-1]), 'mbe_n_tuples_inc')

            # print mbe header
            print(mbe_header(exp.order, exp.n_tuples['calc'][-1], \
                             1. if exp.order < calc.thres['start'] else calc.thres['perc']))

            # main mbe function
            hashes_win, inc_win, tot, mean_ndets, min_ndets, max_ndets, \
                mean_inc, min_inc, max_inc = mbe_main(mpi, mol, calc, exp)

            # append window to hashes
            if len(exp.prop[calc.target_mbe]['hashes']) == len(exp.n_tuples['inc']):
                exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win
            else:
                exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

            # append window to increments
            if len(exp.prop[calc.target_mbe]['inc']) == len(exp.n_tuples['inc']):
                exp.prop[calc.target_mbe]['inc'][-1] = inc_win
            else:
                exp.prop[calc.target_mbe]['inc'].append(inc_win)

            # append total property
            exp.prop[calc.target_mbe]['tot'].append(tot)
            if exp.order > exp.min_order:
                exp.prop[calc.target_mbe]['tot'][-1] += exp.prop[calc.target_mbe]['tot'][-2]

            # append determinant statistics
            if len(exp.mean_ndets) > exp.order - exp.min_order:
                exp.mean_ndets[-1] = mean_ndets
                exp.min_ndets[-1] = min_ndets
                exp.max_ndets[-1] = max_ndets
            else:
                exp.mean_ndets.append(mean_ndets)
                exp.min_ndets.append(min_ndets)
                exp.max_ndets.append(max_ndets)

            # append increment statistics
            if len(exp.mean_inc) > exp.order - exp.min_order:
                exp.mean_inc[-1] = mean_inc
                exp.min_inc[-1] = min_inc
                exp.max_inc[-1] = max_inc
            else:
                exp.mean_inc.append(mean_inc)
                exp.min_inc.append(min_inc)
                exp.max_inc.append(max_inc)

            # print mbe end
            print(mbe_end(exp.order, exp.time['mbe'][-1]))

            # print mbe results
            print(mbe_results(calc.target_mbe, calc.state['root'], exp.min_order, \
                              exp.order, exp.prop[calc.target_mbe]['tot'], \
                              exp.mean_inc[-1], exp.min_inc[-1], exp.max_inc[-1], \
                              exp.mean_ndets[-1], exp.min_ndets[-1], exp.max_ndets[-1]))

            # train ml model
            if (exp.order == calc.thres['start']):
                model = train(mol, calc, exp)

            # update screen_orbs
            if exp.order == exp.min_order:
                exp.screen_orbs = np.array([], dtype=np.int64)
            else:
                exp.screen_orbs = np.setdiff1d(exp.exp_space[-2], exp.exp_space[-1])

            # print screening results
            if 0 < exp.screen_orbs.size:
                print(screen_results(exp.order, exp.screen_orbs, exp.exp_space))

            # print header
            print(purge_header(exp.order))

            # main purging function
            exp.prop[calc.target_mbe], exp.n_tuples = purge_main(mpi, mol, calc, exp)

            # print purging results
            if exp.order + 1 <= exp.exp_space[-1].size:
                print(purge_results(exp.n_tuples, exp.min_order, exp.order))

            # print purge end
            print(purge_end(exp.order, exp.time['purge'][-1]))

            # write restart files
            if calc.misc['rst']:
                if exp.screen_orbs.size > 0:
                    for k in range(exp.order-exp.min_order+1):
                        buf = exp.prop[calc.target_mbe]['hashes'][k].Shared_query(0)[0] # type: ignore
                        hashes = np.ndarray(buffer = buf, dtype=np.int64, \
                                            shape = (exp.n_tuples['inc'][k],))
                        write_file(k + exp.min_order, hashes, 'mbe_hashes')
                        buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
                        inc = np.ndarray(buffer=buf, dtype=np.float64, \
                                         shape = inc_shape(exp.n_tuples['inc'][k], inc_dim(calc.target_mbe)))
                        write_file(k + exp.min_order, inc, 'mbe_inc')
                        write_file(k + exp.min_order, np.asarray(exp.n_tuples['inc'][k]), 'mbe_n_tuples_inc')
                else:
                    buf = exp.prop[calc.target_mbe]['hashes'][-1].Shared_query(0)[0] # type: ignore
                    hashes = np.ndarray(buffer=buf, dtype=np.int64, \
                                        shape=(exp.n_tuples['inc'][-1],))
                    write_file(exp.order, hashes, 'mbe_hashes')
                    buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
                    inc = np.ndarray(buffer=buf, dtype=np.float64, \
                                     shape = inc_shape(exp.n_tuples['inc'][-1], inc_dim(calc.target_mbe)))
                    write_file(exp.order, inc, 'mbe_inc')
                    write_file(exp.order, np.asarray(exp.n_tuples['inc'][-1]), 'mbe_n_tuples_inc')
                write_file(exp.order, np.asarray(exp.prop[calc.target_mbe]['tot'][-1]), 'mbe_tot')
                write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                write_file(exp.order, np.asarray(exp.time['purge'][-1]), 'mbe_time_purge')

            # convergence check
            if exp.exp_space[-1].size < exp.order + 1 or exp.order == exp.max_order:

                # final order
                exp.final_order = exp.order

                # timings
                exp.time['mbe'] = np.asarray(exp.time['mbe'])
                exp.time['purge'] = np.asarray(exp.time['purge'])
                exp.time['total'] = exp.time['mbe'] + exp.time['purge']

                # increments
                exp.mean_inc = np.asarray(exp.mean_inc)
                exp.min_inc = np.asarray(exp.min_inc)
                exp.max_inc = np.asarray(exp.max_inc)

                # ndets
                exp.mean_ndets = np.asarray(exp.mean_ndets)
                exp.min_ndets = np.asarray(exp.min_ndets)
                exp.max_ndets = np.asarray(exp.max_ndets)

                # final results
                exp.prop[calc.target_mbe]['tot'] = np.asarray(exp.prop[calc.target_mbe]['tot'])
                print('\n\n')

                break


def slave(mpi: MPICls, mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
        """
        this function is the main pymbe slave function
        """
        # set loop/waiting logical
        slave = True

        # enter slave state
        while slave:

            # task id
            msg = mpi.global_comm.bcast(None, root=0)

            if msg['task'] == 'mbe':

                # receive order
                exp.order = msg['order']

                # actual number of tuples at current order
                if len(exp.n_tuples['inc']) == exp.order - exp.min_order:
                    exp.n_tuples['inc'].append(n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                                         exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                                         occ_prune(calc.occup, calc.ref_space), \
                                                         virt_prune(calc.occup, calc.ref_space), exp.order))

                # main mbe function
                hashes_win, inc_win = mbe_main(mpi, mol, calc, exp, \
                                               rst_read=msg['rst_read'], \
                                               tup_idx=msg['tup_idx'], \
                                               tup=msg['tup'])

                # append window to hashes
                if len(exp.prop[calc.target_mbe]['hashes']) == len(exp.n_tuples['inc']):
                    exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win
                else:
                    exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

                # append window to increments
                if len(exp.prop[calc.target_mbe]['inc']) == len(exp.n_tuples['inc']):
                    exp.prop[calc.target_mbe]['inc'][-1] = inc_win
                else:
                    exp.prop[calc.target_mbe]['inc'].append(inc_win)

                # update screen_orbs
                if exp.order == exp.min_order:
                    exp.screen_orbs = np.array([], dtype=np.int64)
                else:
                    exp.screen_orbs = np.setdiff1d(exp.exp_space[-2], exp.exp_space[-1])

            elif msg['task'] == 'purge':

                # receive order
                exp.order = msg['order']

                # main purging function
                exp.prop[calc.target_mbe], exp.n_tuples = purge_main(mpi, mol, calc, exp)

            elif msg['task'] == 'exit':

                slave = False

        # finalize mpi
        mpi_finalize(mpi, calc.misc['rst'])


