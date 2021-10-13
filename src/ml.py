#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
machine learning module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'


import numpy as np
from sklearn.kernel_ridge import KernelRidge

from system import MolCls
from calculation import CalcCls
from expansion import ExpCls
from tools import inc_dim, inc_shape, occ_prune, virt_prune, tuples, \
                  hash_lookup, hash_1d


def train(mol: MolCls, calc: CalcCls, exp: ExpCls) -> KernelRidge:
        """
        this function trains the ML model
        """
        # increment dimensions
        dim = inc_dim(calc.target_mbe)

        # load hashes for previous orders
        hashes = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.prop[calc.target_mbe]['hashes'][k].Shared_query(0)[0] # type: ignore
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k],)))

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][k], dim)))

        # allow for tuples with only virtual or occupied MOs
        ref_occ = occ_prune(calc.occup, calc.ref_space)
        ref_virt = virt_prune(calc.occup, calc.ref_space)

        # allocate space for training
        x = np.zeros((sum(exp.n_tuples['calc']), exp.exp_space[0].size), dtype=np.int64)
        y = np.empty(sum(exp.n_tuples['calc']), dtype=np.float64)

        tup_idx = 0

        for order in range(exp.min_order, exp.order+1):

            # occupied and virtual expansion spaces
            exp_occ = exp.exp_space[order-exp.min_order][exp.exp_space[order-exp.min_order] < mol.nocc]
            exp_virt = exp.exp_space[order-exp.min_order][mol.nocc <= exp.exp_space[order-exp.min_order]]

            for tup_idx, tup in enumerate(tuples(exp_occ, exp_virt, ref_occ, \
                                          ref_virt, order), tup_idx):

                # set mo features
                for orb in tup:
                    x[tup_idx, np.where(exp.exp_space[0] == orb)] = 1

                # compute index
                idx = hash_lookup(hashes[order-exp.min_order], hash_1d(tup))

                # set increments
                y[tup_idx] = np.abs(inc[order-exp.min_order][idx])

            tup_idx += 1

        np.set_printoptions(threshold=np.inf)

        clf = KernelRidge(alpha=1.0)
        clf.fit(x, y)

        return clf
