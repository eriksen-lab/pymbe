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
from typing import List, Generator
from itertools import islice, combinations
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PowerTransformer

from system import MolCls
from calculation import CalcCls
from expansion import ExpCls
from tools import inc_dim, inc_shape, occ_prune, virt_prune, tuples, \
                  hash_lookup, hash_1d, core_cas, idx_tril
from kernel import e_core_h1e
from mbe import _inc, _sum


class MLCls:
        """
        this class contains the ML attributes
        """
        def __init__(self) -> None:
                """
                init ML attributes
                """
                self.model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
                self.transformer = PowerTransformer(method='box-cox')
                self.x_train: List[np.ndarray] = []
                self.y_train: List[np.ndarray] = []


        def add_data(self, mol: MolCls, calc: CalcCls, exp: ExpCls) -> None:
                """
                add training data
                """
                # increment dimensions
                dim = inc_dim(calc.target_mbe)

                # load hashes for this order
                buf = exp.prop[calc.target_mbe]['hashes'][-1].Shared_query(0)[0] # type: ignore
                hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['calc'][-1],))

                # load increments for this order
                buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['calc'][-1], dim))

                # allow for tuples with only virtual or occupied MOs
                ref_occ = occ_prune(calc.occup, calc.ref_space)
                ref_virt = virt_prune(calc.occup, calc.ref_space)

                # allocate space for training
                mo = np.zeros((exp.n_tuples['calc'][-1], exp.exp_space[0].size), dtype=np.int64)
                order = np.empty(exp.n_tuples['calc'][-1], dtype=np.int64)
                y = np.empty(exp.n_tuples['calc'][-1], dtype=np.float64)

                tup_idx = 0

                # occupied and virtual expansion spaces
                exp_occ = exp.exp_space[-2][exp.exp_space[-2] < mol.nocc]
                exp_virt = exp.exp_space[-2][mol.nocc <= exp.exp_space[-2]]

                for tup_idx, tup in enumerate(tuples(exp_occ, exp_virt, ref_occ, \
                                                     ref_virt, exp.order)):

                    # set mo features
                    for orb in tup:
                        mo[tup_idx, np.where(exp.exp_space[0] == orb)] = 1

                    # set order
                    order[tup_idx] = exp.order

                    # compute index
                    idx = hash_lookup(hashes, hash_1d(tup))

                    # set increments
                    y[tup_idx] = inc[idx]

                x = np.concatenate((order[:, np.newaxis], mo), axis=1)
                self.x_train.append(x)
                self.y_train.append(y)


        def train(self) -> None:
                """
                this function trains the ML model
                """
                print('training model')

                x = np.concatenate(self.x_train)
                y = np.abs(np.concatenate(self.y_train)).reshape(-1, 1)
                y = self.transformer.fit_transform(y)

                self.model.fit(x, y)
                print('Score on training set:', self.model.score(x, y))

                #param_grid = [
                #{'gamma': [1.e-2, 1.e-1, 1., 1.e1, 1.e2], 'alpha': [1., 1.e-1, 1.e-2, 1.e-3]}
                #]

                #cv = GridSearchCV(self.model, param_grid)
                
                #cv.fit(x, y)
                #print(cv.cv_results_)
                #print(cv.best_estimator_)


        def predict(self, mol: MolCls, calc: CalcCls, exp: ExpCls, \
                    hashes: List[np.ndarray], inc: List[np.ndarray]) -> None:
                """
                this function predicts with the ML model
                """
                np.set_printoptions(threshold=np.inf)
                # load eri
                buf = mol.eri.Shared_query(0)[0]
                eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

                # load hcore
                buf = mol.hcore.Shared_query(0)[0]
                hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

                # load vhf
                buf = mol.vhf.Shared_query(0)[0]
                vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

                # define arrays for different orbital types
                new_exp_occ = exp.exp_space[-2][exp.exp_space[-2] < mol.nocc]
                new_exp_virt = exp.exp_space[-2][mol.nocc <= exp.exp_space[-2]]

                old_exp_occ = exp.exp_space[-3][exp.exp_space[-3] < mol.nocc]
                old_exp_virt = exp.exp_space[-3][mol.nocc <= exp.exp_space[-3]]

                screened_occ = np.setdiff1d(old_exp_occ, new_exp_occ)
                screened_virt = np.setdiff1d(old_exp_virt, new_exp_virt)

                ref_occ = occ_prune(calc.occup, calc.ref_space)
                ref_virt = virt_prune(calc.occup, calc.ref_space)

                features = []
                pred_incs = []
                calc_incs = []

                for tup in screened_tuples(screened_occ, screened_virt, new_exp_occ, new_exp_virt, ref_occ, ref_virt, exp.order):

                    mo = np.zeros(exp.exp_space[0].size, dtype=np.int64)

                    # set mo features
                    for orb in tup:
                        mo[np.where(exp.exp_space[0] == orb)] = 1

                    # set order
                    order = exp.order

                    features.append(np.concatenate(([order], mo)))

                    pred_incs.append(self.transformer.inverse_transform(self.model.predict(features[-1].reshape(1, -1))))

                    # get core and cas indices
                    core_idx, cas_idx = core_cas(mol.nocc, calc.ref_space, tup)

                    # get h2e indices
                    cas_idx_tril = idx_tril(cas_idx)

                    # get h2e_cas
                    h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                    # compute e_core and h1e_cas
                    e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                    # calculate increment
                    calc_inc, _, _ = _inc(calc.model, calc.base['method'], calc.orbs['type'], mol.spin, \
                                                    calc.occup, calc.target_mbe, calc.state, mol.groupname, \
                                                    calc.orbsym, calc.prop, e_core, h1e_cas, h2e_cas, core_idx, \
                                                    cas_idx, mol.debug, mol.dipole_ints)

                    # calculate increment
                    if exp.order > exp.min_order:
                        calc_inc -= _sum(mol.nocc, calc.target_mbe, exp.min_order, exp.order, \
                                         inc, hashes, ref_occ, ref_virt, tup)

                    calc_incs.append(calc_inc)

                    print(tup, pred_incs[-1], calc_incs[-1])

                features = np.array(features)
                calc_incs = self.transformer.transform(np.array(calc_incs).reshape(-1, 1))

                print('Score on testing set:', self.model.score(features, calc_incs))

                exit()



def screened_tuples(screen_occ_space: np.ndarray, screen_virt_space: np.ndarray, \
                    noscreen_occ_space: np.ndarray, noscreen_virt_space: np.ndarray, \
                    ref_occ: bool, ref_virt: bool, order: int) -> Generator[np.ndarray, None, None]:
        """
        this function is the main generator for screened tuples
        """
        # loop over number of screened orbitals in tuple, at least one screened orbital has to be present for a tuple to be screened out
        for screened_orbs in range(1, order + 1):

            # loop over number of screened occupied orbitals in tuple
            for k in range(0, screened_orbs + 1):

                # loop over number of unscreened occupied orbitals in this tuple, k + l must be at least 1 and at most next_order - screened_orbs - 1 if ref_virt or ref_occ are not True
                for l in range(0 if k > 0 else 1, order - screened_orbs + 1 if k < screened_orbs else order - screened_orbs):
                    
                    for tup_screen_occ in islice(combinations(screen_occ_space, k), None):
                        for tup_screen_virt in islice(combinations(screen_virt_space, screened_orbs - k), None):
                            for tup_noscreen_occ in islice(combinations(noscreen_occ_space, l), None):
                                for tup_noscreen_virt in islice(combinations(noscreen_virt_space, order - screened_orbs - l), None):

                                    tup = np.array(tup_screen_occ + tup_screen_virt + tup_noscreen_occ + tup_noscreen_virt, dtype=np.int64)
                                    tup.sort()

                                    yield tup

            # only occupied MOs
            if ref_virt:

                for tup_screen_occ in islice(combinations(screen_occ_space, screened_orbs), None):
                    for tup_noscreen_occ in islice(combinations(noscreen_occ_space, order - screened_orbs), None):

                        tup = np.array(tup_screen_occ + tup_noscreen_occ, dtype=np.int64)
                        tup.sort()

                        yield tup

            # only virtual MOs
            if ref_occ:
                
                for tup_screen_virt in islice(combinations(screen_virt_space, screened_orbs), None):
                    for tup_noscreen_virt in islice(combinations(noscreen_virt_space, order - screened_orbs), None):

                        tup = np.array(tup_screen_virt + tup_noscreen_virt, dtype=np.int64)
                        tup.sort()

                        yield tup

