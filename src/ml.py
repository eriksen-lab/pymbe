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
from mpi4py import MPI
from typing import List, Generator, Tuple, Any, Union
from random import shuffle, randrange, sample, seed
from itertools import islice
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from system import MolCls
from calculation import CalcCls
from expansion import ExpCls
from parallel import MPICls, mpi_reduce, mpi_allreduce
from tools import inc_dim, inc_shape, occ_prune, virt_prune, tuples, \
                  hash_lookup, hash_1d, core_cas, idx_tril, fsum
from kernel import e_core_h1e
from mbe import _inc, _sum, _update
from output import mbe_debug, mbe_status


# seed random number generators
SEED = 0

# number of samples to use at current order
TRAIN_SIZE = 1000

# batch size for SGD algorithm
BATCH_SIZE = 128

# number of epochs to train for
N_EPOCHS = 10000

# number of strata to use
N_STRATA = 6

# seeds
seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rc('text', **{'usetex': True})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

sns.set_theme()

class MLCls():
        """
        this class contains the ML attributes
        """
        def __init__(self, n_features: int, min_order: int) -> None:
                """
                init ML attributes
                """
                # set maximum number of increments to calculate at every order
                self.max_calcs = 5000

                # set number of features
                self.n_features = n_features

                # define the network and print architecture
                self.net = Net(n_feature=self.n_features, n_hidden=15, n_output=1)
                print(self.net)

                # define optimizer
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)

                # define loss function
                self.loss_func = torch.nn.MSELoss()

                # set patience before stopping
                self.patience = 10

                # set minimum delta for loss improvement
                self.min_delta = 0.

                # increments of all orders need to be added to training set
                self.last_order = min_order

                # initialize training set
                self.x_train: List[np.ndarray] = []
                self.y_train: List[np.ndarray] = []

                self.strata_ranges = np.logspace(-1, -10, N_STRATA)

        def fit_transform(self, input: np.ndarray) -> np.ndarray:

                # define transformer
                self.transformer = StandardScaler()

                return self.transformer.fit_transform(np.log(input + 5.e-11))


        def transform(self, input: np.ndarray) -> np.ndarray:

                return self.transformer.transform(np.log(input + 5.e-11))


        def inverse_transform(self, input: np.ndarray) -> np.ndarray:

                return np.exp(self.transformer.inverse_transform(input)) - 5.e-11


        def add_data(self, mol: MolCls, calc: CalcCls, exp: ExpCls, hashes: List[np.ndarray], inc: List[np.ndarray]) -> None:
                """
                this function adds training data from previous orders
                """
                # allow for tuples with only virtual or occupied MOs
                ref_occ = occ_prune(calc.occup, calc.ref_space)
                ref_virt = virt_prune(calc.occup, calc.ref_space)

                # occupied and virtual expansion spaces
                exp_occ = exp.exp_space[0][exp.exp_space[0] < mol.nocc]
                exp_virt = exp.exp_space[0][mol.nocc <= exp.exp_space[0]]

                #fao = calc.hf.get_fock()
                #fmo = calc.mo_coeff.T @ fao @ calc.mo_coeff
                #mocc = calc.mo_coeff[:,calc.occup>0]
                #dm = np.dot(mocc*calc.occup[calc.occup>0], mocc.T)
                #jmo, kmo = calc.hf.get_jk(mol, dm)

                for k in range(self.last_order, exp.order):

                    i = k - exp.min_order

                    # allocate space for training
                    mo = np.zeros((exp.n_tuples['inc'][i], exp.exp_space[0].size), dtype=np.float64)
                    #qm = np.zeros((exp.n_tuples['inc'][i], 12), dtype=np.float64)
                    y = np.empty(exp.n_tuples['inc'][i], dtype=np.float64)

                    tup_idx = 0

                    for tup in tuples(exp_occ, exp_virt, ref_occ, ref_virt, k):

                        # compute index
                        idx = hash_lookup(hashes[i], hash_1d(tup))

                        if idx is not None:

                            # set mo features
                            for orb in tup:
                                mo[tup_idx, np.where(exp.exp_space[0] == orb)] = 1.

                            #fock_tuples = fmo[tup[:, np.newaxis], tup]
                            #coulomb_tuples = jmo[tup[:, np.newaxis], tup]
                            #exchange_tuples = kmo[tup[:, np.newaxis], tup]

                            #fock_off_diag = fock_tuples[np.triu_indices_from(fock_tuples, k=1)]
                            #coulomb_off_diag = coulomb_tuples[np.triu_indices_from(coulomb_tuples, k=1)]
                            #exchange_off_diag = exchange_tuples[np.triu_indices_from(exchange_tuples, k=1)]

                            #qm[tup_idx, 0] = np.max(np.abs(coulomb_off_diag))
                            #qm[tup_idx, 1] = np.min(np.abs(coulomb_off_diag))
                            #qm[tup_idx, 2] = np.mean(coulomb_off_diag)
                            #qm[tup_idx, 3] = np.mean(np.abs(coulomb_off_diag))
                            #qm[tup_idx, 4] = np.std(coulomb_off_diag)
                            #qm[tup_idx, 5] = np.std(np.abs(coulomb_off_diag))
                            #qm[tup_idx, 6] = np.max(np.abs(exchange_off_diag))
                            #qm[tup_idx, 7] = np.min(np.abs(exchange_off_diag))
                            #qm[tup_idx, 8] = np.mean(exchange_off_diag)
                            #qm[tup_idx, 9] = np.mean(np.abs(exchange_off_diag))
                            #qm[tup_idx, 10] = np.std(exchange_off_diag)
                            #qm[tup_idx, 11] = np.std(np.abs(exchange_off_diag))

                            # set increments
                            y[tup_idx] = inc[i][idx]
                            tup_idx += 1

                    # only append data to training set if no data has been added yet for this order
                    if len(self.x_train) < self.last_order:

                        self.x_train.append(mo)
                        #self.x_train.append(np.concatenate((mo, qm), axis=1))
                        self.y_train.append(y)

                    else:

                        self.x_train[-1] = mo
                        self.y_train[-1] = y

                    self.last_order = k


        def sample_order(self, mol: MolCls, calc: CalcCls, exp: ExpCls, exp_occ: np.ndarray, 
                         exp_virt: np.ndarray, ref_occ: bool, ref_virt: bool, eri: np.ndarray,
                         hcore: np.ndarray, vhf: np.ndarray, hashes: List[np.ndarray],
                         inc: List[np.ndarray]):
                """
                this function samples random tuples from this order and adds these to training data
                """
                # generate random tuples for this order
                random_tuples = tuple(random_sample(avail_tuples(tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order), exp.order, exp.min_order, mol.nocc, ref_occ, ref_virt, hashes, inc), TRAIN_SIZE))

                # allocate space for training 
                mo = np.zeros((TRAIN_SIZE, self.n_features), dtype=np.float64)
                y = np.empty(TRAIN_SIZE, dtype=np.float64)

                if len(random_tuples) == TRAIN_SIZE:

                    for tup_idx, (tup, subtup_sum) in enumerate(random_tuples):

                        for orb in tup:
                            mo[tup_idx, np.where(exp.exp_space[0] == orb)] = 1.

                        # get core and cas indices
                        core_idx, cas_idx = core_cas(mol.nocc, calc.ref_space, tup)

                        # get h2e indices
                        cas_idx_tril = idx_tril(cas_idx)

                        # get h2e_cas
                        h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                        # compute e_core and h1e_cas
                        e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                        # calculate increment
                        y[tup_idx], _, _ = _inc(calc.model, calc.base['method'], calc.orbs['type'], mol.spin, \
                                                calc.occup, calc.target_mbe, calc.state, mol.groupname, \
                                                calc.orbsym, calc.prop, e_core, h1e_cas, h2e_cas, \
                                                core_idx, cas_idx, mol.debug, mol.dipole_ints)

                        # calculate increment
                        y[tup_idx] -= subtup_sum

                    self.x_train.append(mo)
                    self.y_train.append(y)


        def validate(self, mol: MolCls, calc: CalcCls, exp: ExpCls, exp_occ: np.ndarray, 
                           exp_virt: np.ndarray, ref_occ: bool, ref_virt: bool, eri: np.ndarray,
                           hcore: np.ndarray, vhf: np.ndarray, hashes: List[np.ndarray],
                           inc: List[np.ndarray]):
                """
                this function samples random tuples from this order and adds these to training data
                """
                #fao = calc.hf.get_fock()
                #fmo = calc.mo_coeff.T @ fao @ calc.mo_coeff
                #mocc = calc.mo_coeff[:,calc.occup>0]
                #dm = np.dot(mocc*calc.occup[calc.occup>0], mocc.T)
                #jmo, kmo = calc.hf.get_jk(mol, dm)

                self.x_valid = np.zeros((exp.n_tuples['theo'][-1], self.n_features), dtype=np.float64)
                self.y_valid = np.empty(exp.n_tuples['theo'][-1], dtype=np.float64)

                for tup_idx, tup in enumerate(tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order)):

                    # set mo features
                    for orb in tup:
                        self.x_valid[tup_idx, np.where(exp.exp_space[0] == orb)] = 1.

                    #fock_tuples = fmo[tup[:, np.newaxis], tup]
                    #coulomb_tuples = jmo[tup[:, np.newaxis], tup]
                    #exchange_tuples = kmo[tup[:, np.newaxis], tup]

                    #fock_off_diag = fock_tuples[np.triu_indices_from(fock_tuples, k=1)]
                    #coulomb_off_diag = coulomb_tuples[np.triu_indices_from(coulomb_tuples, k=1)]
                    #exchange_off_diag = exchange_tuples[np.triu_indices_from(exchange_tuples, k=1)]

                    #self.x_valid[tup_idx, exp.exp_space[0].size] = np.max(np.abs(coulomb_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+1] = np.min(np.abs(coulomb_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+2] = np.mean(coulomb_off_diag)
                    #self.x_valid[tup_idx, exp.exp_space[0].size+3] = np.mean(np.abs(coulomb_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+4] = np.std(coulomb_off_diag)
                    #self.x_valid[tup_idx, exp.exp_space[0].size+5] = np.std(np.abs(coulomb_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+6] = np.max(np.abs(exchange_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+7] = np.min(np.abs(exchange_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+8] = np.mean(exchange_off_diag)
                    #self.x_valid[tup_idx, exp.exp_space[0].size+9] = np.mean(np.abs(exchange_off_diag))
                    #self.x_valid[tup_idx, exp.exp_space[0].size+10] = np.std(exchange_off_diag)
                    #self.x_valid[tup_idx, exp.exp_space[0].size+11] = np.std(np.abs(exchange_off_diag))

                    # get core and cas indices
                    core_idx, cas_idx = core_cas(mol.nocc, calc.ref_space, tup)

                    # get h2e indices
                    cas_idx_tril = idx_tril(cas_idx)

                    # get h2e_cas
                    h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                    # compute e_core and h1e_cas
                    e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                    # calculate increment
                    self.y_valid[tup_idx], _, _ = _inc(calc.model, calc.base['method'], calc.orbs['type'], mol.spin, \
                                                       calc.occup, calc.target_mbe, calc.state, mol.groupname, \
                                                       calc.orbsym, calc.prop, e_core, h1e_cas, h2e_cas, \
                                                       core_idx, cas_idx, mol.debug, mol.dipole_ints)

                    # calculate increment
                    if exp.order > exp.min_order:
                        self.y_valid[tup_idx] -= _sum(mol.nocc, calc.target_mbe, exp.min_order, exp.order, \
                                                      inc, hashes, ref_occ, ref_virt, tup)

                print('training model')

                fig, ax = plt.subplots(figsize=(8, 5))

                percentages = [0, 1, 2, 5, 10, 20, 50]

                for idx, percent in enumerate(percentages):

                    # reset model
                    for layer in self.net.children():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()

                    random_tuples = sample(range(0, self.y_valid.size), int(round(percent / 100. * self.y_valid.size)))

                    self.x_train.append(self.x_valid[random_tuples, :])
                    self.y_train.append(self.y_valid[random_tuples])

                    x_train = np.concatenate(self.x_train)
                    y_train = np.abs(np.concatenate(self.y_train)).reshape(-1, 1)
                    #y_train[y_train < 1.e-10] = 1.e-10
                    y_train = self.fit_transform(y_train)

                    x_train, y_train = map(torch.as_tensor, (x_train, y_train))

                    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                            batch_size=BATCH_SIZE, 
                                                            shuffle=True, num_workers=0)

                    x_valid = np.delete(self.x_valid, random_tuples, axis=0)
                    y_valid = np.abs(np.delete(self.y_valid, random_tuples)).reshape(-1, 1)
                    #y_valid[y_valid < 1.e-10] = 1.e-10
                    y_valid = self.transform(y_valid)

                    x_valid, y_valid = map(torch.as_tensor, (x_valid, y_valid))

                    valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)

                    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                                            batch_size=BATCH_SIZE, 
                                                            shuffle=True, num_workers=0)

                    train_loss = []
                    valid_loss = []
                    best_loss: float

                    # train the network
                    for epoch in range(1, N_EPOCHS+1):

                        train_loss.append(0.0)

                        # loop over batches
                        for x_batch, y_batch in train_loader:
                    
                            # predict y based on x
                            prediction = self.net(x_batch)

                            # calculate loss in comparison to training data
                            loss = self.loss_func(prediction, y_batch)

                            # clear gradients
                            self.optimizer.zero_grad()
                            
                            # backpropagation, compute gradients
                            loss.backward()

                            # add loss to epoch loss
                            train_loss[-1] += loss.item() * x_batch.size(0)

                            # apply gradients
                            self.optimizer.step()

                        valid_loss.append(0.0)

                        # loop over batches
                        for x_batch, y_batch in valid_loader:
                            
                            # predict y based on x
                            prediction = self.net(x_batch)

                            # calculate loss in comparison to training data
                            loss = self.loss_func(prediction, y_batch)

                            # add loss to epoch loss
                            valid_loss[-1] += loss.item() * x_batch.size(0)

                        train_loss[-1] /= len(train_loader)
                        valid_loss[-1] /= len(valid_loader)

                        print("epoch = %4d   training loss = %0.4f   validation loss = %0.4f" % (epoch, train_loss[-1], valid_loss[-1]))

                        # check if loss has not improved by min_delta for patience epochs
                        if epoch == 1 or best_loss - train_loss[-1] > self.min_delta:

                            best_loss = train_loss[-1]

                            counter = 0

                        else:

                            counter += 1
                            
                            if counter >= self.patience:

                                break

                    epochs = np.arange(1, epoch+1)

                    ax.plot(epochs, train_loss, color='C' + str(idx), linewidth=1, linestyle='solid', label='Training loss')
                    ax.plot(epochs, valid_loss, color='C' + str(idx), linewidth=1, linestyle='dashed', label='Validation loss')

                    self.x_train.pop()
                    self.y_train.pop()

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')

                # dummy plots
                d1,  = ax.plot([0], marker='None', linestyle='None', label='dummy-tophead')
                d2,  = ax.plot([0], color='C0', linewidth=1, marker='None', label=r'$0\%$')
                d3,  = ax.plot([0], color='C1', linewidth=1, marker='None', label=r'$1\%$')
                d4,  = ax.plot([0], color='C2', linewidth=1, marker='None', label=r'$2\%$')
                d5,  = ax.plot([0], color='C3', linewidth=1, marker='None', label=r'$5\%$')
                d6,  = ax.plot([0], color='C4', linewidth=1, marker='None', label=r'$10\%$')
                d7,  = ax.plot([0], color='C5', linewidth=1, marker='None', label=r'$20\%$')
                d8,  = ax.plot([0], color='C6', linewidth=1, marker='None', label=r'$50\%$')
                d9,  = ax.plot([0], color='black', linewidth=1, marker='None', linestyle='solid', label='Training loss')
                d10,  = ax.plot([0], color='black', linewidth=1, marker='None', linestyle='dashed', label='Validation loss')

                ax.legend([d1, d2, d3, d4, d5, d6, d7, d8, d1, d9, d10], ('Current order training size', d2.get_label(), d3.get_label(), d4.get_label(), d5.get_label(), d6.get_label(), d7.get_label(), d8.get_label(), 'Loss', d9.get_label(), d10.get_label()), loc='upper right')

                fig.savefig('learning_curve.pdf', bbox_inches='tight')

                y_calc = np.abs(np.delete(self.y_valid, random_tuples))
                                                
                # predict with neural net
                y_pred = self.net(x_valid)

                # convert to numpy array
                y_pred = y_pred.detach().numpy().reshape(-1, 1)

                # backtransform prediction
                y_pred = self.inverse_transform(y_pred)

                fig, ax = plt.subplots(figsize=(8, 5))

                for i in range(1, self.strata_ranges.size):

                    rect = patches.Rectangle((self.strata_ranges[i], self.strata_ranges[i]), self.strata_ranges[i-1]-self.strata_ranges[i], self.strata_ranges[i-1]-self.strata_ranges[i], facecolor='black', alpha=0.3)
                    ax.add_patch(rect)

                ax.scatter(y_calc, y_pred, s=1, marker='.')

                ax.set_xlabel('Actual increment')
                ax.set_ylabel('Predicted increment')
                ax.set_xscale('log')
                ax.set_yscale('log')

                xlim_left, xlim_right = ax.get_xlim()
                ylim_bot, ylim_top = ax.get_ylim()

                ax.set_xlim(min(xlim_left, ylim_bot), max(xlim_right, ylim_top))
                ax.set_ylim(min(xlim_left, ylim_bot), max(xlim_right, ylim_top))

                fig.savefig('nn_prediction.pdf', bbox_inches='tight')

                exit()


        def train(self) -> None:
                """
                this function trains the ML model
                """
                print('training model')

                fig, ax = plt.subplots(figsize=(8, 5))

                x_train = np.concatenate(self.x_train)
                y_train = np.abs(np.concatenate(self.y_train)).reshape(-1, 1)
                #y_train[y_train < 1.e-10] = 1.e-10
                y_train = self.fit_transform(y_train)

                x_train, y_train = map(torch.as_tensor, (x_train, y_train))

                train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                        batch_size=BATCH_SIZE, 
                                                        shuffle=True, num_workers=0)

                train_loss = []
                best_loss: float

                # train the network
                for epoch in range(1, N_EPOCHS+1):

                    train_loss.append(0.0)

                    # loop over batches
                    for x_batch, y_batch in train_loader:
                
                        # predict y based on x
                        prediction = self.net(x_batch)

                        # calculate loss in comparison to training data
                        loss = self.loss_func(prediction, y_batch)

                        # clear gradients
                        self.optimizer.zero_grad()
                        
                        # backpropagation, compute gradients
                        loss.backward()

                        # add loss to epoch loss
                        train_loss[-1] += loss.item() * x_batch.size(0)

                        # apply gradients
                        self.optimizer.step()

                    train_loss[-1] /= len(train_loader)

                    # check if loss has not improved by min_delta for patience epochs
                    if epoch == 1 or best_loss - train_loss[-1] > self.min_delta:

                        best_loss = train_loss[-1]

                        counter = 0

                    else:

                        counter += 1
                        
                        if counter >= self.patience:

                            print("epoch = %4d   training loss = %0.4f" % (epoch, train_loss[-1]))

                            break

                    if not epoch % 10:
                    
                        print("epoch = %4d   training loss = %0.4f" % (epoch, train_loss[-1]))

                epochs = np.arange(1, epoch+1)

                ax.plot(epochs, train_loss, color='C0', linewidth=1, linestyle='solid')

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Training loss')

                fig.savefig('learning_curve.pdf', bbox_inches='tight')


        def predict(self, order: int, exp_space: np.ndarray, exp_occ: np.ndarray, exp_virt: np.ndarray, 
                    ref_occ: bool, ref_virt: bool, prev_order_hashes: np.ndarray, nocc: int) -> Tuple[List[List[Union[np.ndarray, None]]], List[int], List[int], np.ndarray]:
                """
                this function predicts with the ML model
                """
                strata_total = N_STRATA * [0]
                strata_tups: List[List[Union[np.ndarray, None]]] = [[] for _ in range(N_STRATA)]
                strata_pred_incs: List[List[Union[float, None]]] = [[] for _ in range(N_STRATA)]

                with torch.no_grad():

                    for tup in tuples(exp_occ, exp_virt, ref_occ, ref_virt, order):

                        mo = np.zeros(self.n_features, dtype=np.float64)

                        # set mo features
                        for orb in tup:
                            mo[np.where(exp_space == orb)] = 1.

                        # concatenate features and convert to torch tensor
                        x = torch.as_tensor(mo)
                                                        
                        # predict with neural net and convert back to ndarray
                        y = self.net(x)

                        # convert to numpy array
                        y = y.detach().numpy()

                        # backtransform prediction
                        y = self.inverse_transform(y).item()

                        i = N_STRATA - np.searchsorted(self.strata_ranges[::-1], y)

                        if i < N_STRATA:
                            
                            strata_total[i] += 1

                            all_subtups_avail = True

                            # occupied and virtual subspaces of tuple
                            tup_occ = tup[tup < nocc]
                            tup_virt = tup[nocc <= tup]

                            # loop over subtuples
                            for tup_sub in tuples(tup_occ, tup_virt, ref_occ, ref_virt, order-1):

                                # compute index
                                idx = hash_lookup(prev_order_hashes, hash_1d(tup_sub))

                                # check if subtuple exists
                                if idx is None:

                                    # if subtuple does not exist predict magnitude
                                    mo = np.zeros(self.n_features, dtype=np.float64)

                                    # set mo features
                                    for orb in tup_sub:
                                        mo[np.where(exp_space == orb)] = 1.

                                    # concatenate features and convert to torch tensor
                                    x = torch.as_tensor(mo)
                                                                    
                                    # predict with neural net and convert back to ndarray
                                    y = self.net(x)

                                    # convert to numpy array
                                    y = y.detach().numpy()

                                    # backtransform prediction
                                    y = self.inverse_transform(y).item()

                                    j = N_STRATA - np.searchsorted(self.strata_ranges[::-1], y)

                                    # check if subtuple is predicted to be numerically relevant
                                    if j < N_STRATA:

                                        all_subtups_avail = False
                                        strata_pred_incs[i].append(y)
                                        break

                            if all_subtups_avail:

                                strata_tups[i].append(tup)

                tot_strata = 0.

                for stratum, stratum_total in enumerate(strata_total):

                    tot_strata += stratum_total * self.strata_ranges[stratum-1]

                strata_samples = []

                n_calcs = self.max_calcs

                for stratum, (stratum_tup, stratum_total) in enumerate(zip(strata_tups, strata_total)):

                    # calculate necessary sample size for minimal total variance in worst-case scenario (all increments in stratum are at upper stratum bound)
                    if tot_strata > 0.:

                        strata_samples.append(int(round(min(n_calcs * stratum_total * self.strata_ranges[stratum-1] / tot_strata, len(stratum_tup)))))

                    else:

                        strata_samples.append(0)

                    if stratum_total > 0:

                        n_calcs -= strata_samples[-1]

                        # prevent too small sample sizes
                        if n_calcs < 10:

                            strata_samples[-1] += n_calcs
                            n_calcs = 0

                        tot_strata -= stratum_total * self.strata_ranges[stratum-1]

                        print('Calculating', strata_samples[-1], 'samples of', stratum_total, 'tuples in stratum', stratum, 'between', self.strata_ranges[stratum-1], 'and', self.strata_ranges[stratum])

                print()

                return strata_tups, strata_total, strata_samples, strata_pred_incs


class Net(torch.nn.Module):
        """
        this class contains a neural net
        """
        def __init__(self, n_feature, n_hidden, n_output) -> None:
                """
                init net attributes
                """
                super(Net, self).__init__()
                self.input = torch.nn.Linear(n_feature, n_hidden)    # input layer
                self.hidden = torch.nn.Linear(n_hidden, n_hidden)    # hidden layer
                self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
                self.double()


        def forward(self, x):
                """
                forward step of net
                """
                x = torch.nn.functional.relu(self.input(x))          # activation function for input layer
                x = torch.nn.functional.relu(self.hidden(x))         # activation function for hidden layer
                x = self.predict(x)                                  # linear output layer
                
                return x


def main(mpi: MPICls, mol: MolCls, calc: CalcCls, exp: ExpCls, ml_object: MLCls, \
         rst_read: bool = False, tup_idx: int = 0, \
         tup: Union[np.ndarray, None] = None) -> Tuple[Any, ...]:
        """
        this function is the mbe main function
        """
        #if mpi.global_master:
            # read restart files
        #    rst_read = is_file(exp.order, 'mbe_idx') and is_file(exp.order, 'mbe_tup')
            # start indices
        #    tup_idx = read_file(exp.order, 'mbe_idx').item() if rst_read else 0
            # start tuples
        #    tup = read_file(exp.order, 'mbe_tup') if rst_read else None
            # wake up slaves
        #    msg = {'task': 'mbe', 'order': exp.order, \
        #           'rst_read': rst_read, 'tup_idx': tup_idx, 'tup': tup}
        #    mpi.global_comm.bcast(msg, root=0)

        # increment dimensions
        dim = inc_dim(calc.target_mbe)

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
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][k], dim)))

        # init time
        if mpi.global_master:
            if not rst_read:
                exp.time['mbe'].append(0.)
            time = MPI.Wtime()

        # init determinant statistics
        min_ndets = exp.min_ndets[-1] if mpi.global_master and rst_read else np.array([1e12], dtype=np.int64)
        max_ndets = exp.max_ndets[-1] if mpi.global_master and rst_read else np.array([0], dtype=np.int64)
        mean_ndets = exp.mean_ndets[-1] if mpi.global_master and rst_read else np.array([0], dtype=np.int64)

        # init increment statistics
        min_inc = exp.min_inc[-1] if mpi.global_master and rst_read else np.array([1.e12] * dim, dtype=np.float64)
        max_inc = exp.max_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)
        mean_inc = exp.mean_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)

        # init pair_corr statistics
        if calc.ref_space.size == 0 and exp.order == exp.min_order and calc.base['method'] is None:
            pair_corr = [np.zeros(exp.n_tuples['inc'][0], dtype=np.float64), \
                         np.zeros([exp.n_tuples['inc'][0], 2], dtype=np.int32)] # type:ignore
        else:
            pair_corr = None # type:ignore

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < mol.nocc]
        exp_virt = exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]]

        # allow for tuples with only virtual or occupied MOs
        ref_occ = occ_prune(calc.occup, calc.ref_space)
        ref_virt = virt_prune(calc.occup, calc.ref_space)

        if exp.n_tuples['inc'][-2] > 0:

            # add data of past orders to ml model
            ml_object.add_data(mol, calc, exp, hashes, inc)

            # validate model
            #ml_object.validate(mol, calc, exp, exp_occ, exp_virt, ref_occ, ref_virt, eri, hcore, vhf, hashes, inc)

            # add sample data from current order to ml model
            ml_object.sample_order(mol, calc, exp, exp_occ, exp_virt, ref_occ, ref_virt, eri, hcore, vhf, hashes, inc)

            # train ml model
            ml_object.train()

        # make predictions with ml model
        strata_tups, strata_total, strata_samples, strata_pred_incs = ml_object.predict(exp.order, exp.exp_space[0], exp_occ, exp_virt, ref_occ, ref_virt, hashes[-1], mol.nocc)

        # actual number of tuples at current order
        exp.n_tuples['calc'][-1] = sum(strata_samples)
        exp.n_tuples['inc'][-1] = exp.n_tuples['calc'][-1]

        # init hashes for present order
        if rst_read:
            hashes_win = exp.prop[calc.target_mbe]['hashes'][-1]
        else:
            hashes_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][-1] if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = hashes_win.Shared_query(0)[0] # type: ignore
        hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][-1],)))
        if mpi.local_master and not mpi.global_master:
            hashes[-1][:].fill(0)

        # init increments for present order
        if rst_read:
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
        else:
            inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][-1] * dim if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = inc_win.Shared_query(0)[0] # type: ignore
        inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][-1], dim)))
        if mpi.local_master and not mpi.global_master:
            inc[-1][:].fill(0.)

        # set rst_write
        #rst_write = calc.misc['rst'] and mpi.global_size < calc.misc['rst_freq'] < exp.n_tuples['inc'][-1]

        # start tuples
        #if tup is not None:
        #    tup_occ = tup[tup < mol.nocc]
        #    tup_virt = tup[mol.nocc <= tup]
        #    if tup_occ.size == 0:
        #        tup_occ = None
        #    if tup_virt.size == 0:
        #        tup_virt = None
        #else:
        #    tup_occ = tup_virt = None
        #order_start, occ_start, virt_start = start_idx(exp_occ, exp_virt, tup_occ, tup_virt)

        energy = 0.
        var = 0.

        # loop until no tuples left
        for stratum, (stratum_tups, stratum_total, stratum_sample) in enumerate(zip(strata_tups, strata_total, strata_samples)):

            prev_tup_idx = tup_idx

            for tup in sample(stratum_tups, stratum_sample):

            # distribute tuples
            #if tup_idx % mpi.global_size != mpi.global_rank:
            #    continue

            # write restart files and re-init time
            #if rst_write and tup_idx % calc.misc['rst_freq'] < mpi.global_size:

                # mpi barrier
            #    mpi.local_comm.Barrier()

                # reduce hashes & increments onto global master
            #    if mpi.num_masters > 1 and mpi.local_master:
            #        hashes[-1][:] = mpi_reduce(mpi.master_comm, hashes[-1], root=0, op=MPI.SUM)
            #        if not mpi.global_master:
            #            hashes[-1][:].fill(0)
            #        inc[-1][:] = mpi_reduce(mpi.master_comm, inc[-1], root=0, op=MPI.SUM)
            #        if not mpi.global_master:
            #            inc[-1][:].fill(0.)

                # reduce increment statistics onto global master
            #    min_inc = mpi_reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
            #    max_inc = mpi_reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
            #    mean_inc = mpi_reduce(mpi.global_comm, mean_inc, root=0, op=MPI.SUM)
            #    if not mpi.global_master:
            #        min_inc = np.array([1.e12] * dim, dtype=np.float64)
            #        max_inc = np.array([0.] * dim, dtype=np.float64)
            #        mean_inc = np.array([0.] * dim, dtype=np.float64)

                # reduce determinant statistics onto global master
            #    min_ndets = mpi_reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
            #    max_ndets = mpi_reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
            #    mean_ndets = mpi_reduce(mpi.global_comm, mean_ndets, root=0, op=MPI.SUM)
            #    if not mpi.global_master:
            #        min_ndets = np.array([1e12], dtype=np.int64)
            #        max_ndets = np.array([0], dtype=np.int64)
            #        mean_ndets = np.array([0], dtype=np.int64)

                # reduce mbe_idx onto global master
            #    mbe_idx = mpi.global_comm.allreduce(tup_idx, op=MPI.MIN)
                # send tup corresponding to mbe_idx to master
            #    if mpi.global_master:
            #        if tup_idx == mbe_idx:
            #            mbe_tup = tup
            #        else:
            #            mbe_tup = np.empty(exp.order, dtype=np.int64)
            #            mpi.global_comm.Recv(mbe_tup, source=MPI.ANY_SOURCE, tag=101)
            #    elif tup_idx == mbe_idx:
            #        mpi.global_comm.Send(tup, dest=0, tag=101)
                # update rst_write
            #    rst_write = mbe_idx + calc.misc['rst_freq'] < exp.n_tuples['inc'][-1] - mpi.global_size

            #    if mpi.global_master:
                    # write restart files
            #        write_file(exp.order, max_inc, 'mbe_max_inc')
            #        write_file(exp.order, min_inc, 'mbe_min_inc')
            #        write_file(exp.order, mean_inc, 'mbe_mean_inc')
            #        write_file(exp.order, max_ndets, 'mbe_max_ndets')
            #        write_file(exp.order, min_ndets, 'mbe_min_ndets')
            #        write_file(exp.order, mean_ndets, 'mbe_mean_ndets')
            #        write_file(exp.order, np.asarray(mbe_idx), 'mbe_idx')
            #        write_file(exp.order, mbe_tup, 'mbe_tup')
            #        write_file(exp.order, hashes[-1], 'mbe_hashes')
            #        write_file(exp.order, inc[-1], 'mbe_inc')
            #        exp.time['mbe'][-1] += MPI.Wtime() - time
            #        write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                    # re-init time
            #        time = MPI.Wtime()
                    # print status
            #        print(mbe_status(exp.order, mbe_idx / exp.n_tuples['inc'][-1]))

                # pi-pruning
            #    if calc.extra['pi_prune']:
            #        if not pi_prune(exp.pi_orbs, exp.pi_hashes, tup):
            #            continue

                # get core and cas indices
                core_idx, cas_idx = core_cas(mol.nocc, calc.ref_space, tup)

                # get h2e indices
                cas_idx_tril = idx_tril(cas_idx)

                # get h2e_cas
                h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                # compute e_core and h1e_cas
                e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                # calculate increment
                inc_tup, ndets_tup, n_elec_tup = _inc(calc.model, calc.base['method'], calc.orbs['type'], mol.spin, \
                                                    calc.occup, calc.target_mbe, calc.state, mol.groupname, \
                                                    calc.orbsym, calc.prop, e_core, h1e_cas, h2e_cas, \
                                                    core_idx, cas_idx, mol.debug, mol.dipole_ints)

                # calculate increment
                if exp.order > exp.min_order:
                    inc_tup -= _sum(mol.nocc, calc.target_mbe, exp.min_order, exp.order, \
                                    inc, hashes, ref_occ, ref_virt, tup)

                # add hash and increment
                hashes[-1][tup_idx] = hash_1d(tup)
                inc[-1][tup_idx] = inc_tup

                # debug print
                if mol.debug >= 2:
                    print(mbe_debug(mol.atom, mol.groupname, calc.orbsym, calc.state['root'], \
                                    ndets_tup, n_elec_tup, inc_tup, exp.order, cas_idx, tup))

                # update increment statistics
                min_inc, max_inc, mean_inc = _update(min_inc, max_inc, mean_inc, inc_tup)
                # update determinant statistics
                min_ndets, max_ndets, mean_ndets = _update(min_ndets, max_ndets, mean_ndets, ndets_tup)
                # update pair_corr statistics
                if pair_corr is not None:
                    if calc.target_mbe in ['energy', 'excitation']:
                        pair_corr[0][tup_idx] = inc_tup # type: ignore
                    else:
                        pair_corr[0][tup_idx] = inc_tup[np.argmax(np.abs(inc_tup))] # type: ignore
                    pair_corr[1][tup_idx] = tup

                tup_idx += 1

            if stratum_total > stratum_sample:
                print('Errors for stratum', stratum)

            if stratum_sample > 0:

                # calculate sample mean of stratum
                sample_mean = fsum(inc[-1][prev_tup_idx:tup_idx]) / stratum_sample

                # estimate population total of all increments in this stratum through sample mean
                energy += len(stratum_tups) * sample_mean

                # variance only necessary if only part of the population was calculated
                if len(stratum_tups) > stratum_sample:

                    # calculate sample variance of stratum
                    sample_var = np.var(inc[-1][prev_tup_idx:tup_idx], ddof=1)

                    # calculate variance of the population total
                    var += len(stratum_tups) * (len(stratum_tups) - stratum_sample) * sample_var / stratum_sample
                    print('Sample variance:', len(stratum_tups) * (len(stratum_tups) - stratum_sample) * sample_var / stratum_sample)

                elif stratum_total > len(stratum_tups):

                    # calculate sample variance for predicted increments in stratum
                    sample_var = np.var(inc[-1][prev_tup_idx:tup_idx], ddof=1)

                    # calculate variance of increments that cannot be constructed due to missing subtuples
                    var += (np.sum(strata_pred_incs[stratum])) ** 2
                    #var += ((stratum_total - len(stratum_tups)) * ml_object.strata_ranges[stratum-1]) ** 2
                    print('Missing subtuple variance:', ((stratum_total - len(stratum_tups)) * ml_object.strata_ranges[stratum-1]) ** 2)

            else:

                # calculate variance of all increments without sample
                var += (np.sum(strata_pred_incs[stratum])) ** 2
                #var += (stratum_total * ml_object.strata_ranges[stratum-1]) ** 2
                if stratum_total > 0:
                    print('Missing subtuple variance:', (stratum_total * ml_object.strata_ranges[stratum-1]) ** 2)

            if stratum_total > stratum_sample:
                print()

        string = u' RESULT-{:d}:  energy = {:.4e} \u00b1 {:.4e}\n'
        form: Tuple[int, float, float] = (exp.order, energy, 2*np.sqrt(var))

        print(string.format(*form))

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            print(mbe_status(exp.order, 1.))

        # allreduce hashes & increments among local masters
        if mpi.local_master:
            hashes[-1][:] = mpi_allreduce(mpi.master_comm, hashes[-1], op=MPI.SUM)
            inc[-1][:] = mpi_allreduce(mpi.master_comm, inc[-1], op=MPI.SUM)

        # sort hashes and increments
        if mpi.local_master:
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1][:].sort()

        # increment statistics
        min_inc = mpi_reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
        max_inc = mpi_reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
        mean_inc = mpi_reduce(mpi.global_comm, mean_inc, root=0, op=MPI.SUM)

        # determinant statistics
        min_ndets = mpi_reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
        max_ndets = mpi_reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
        mean_ndets = mpi_reduce(mpi.global_comm, mean_ndets, root=0, op=MPI.SUM)

        # pair_corr statistics
        if pair_corr is not None:
            pair_corr = [mpi_reduce(mpi.global_comm, pair_corr[0], root=0, op=MPI.SUM), \
                         mpi_reduce(mpi.global_comm, pair_corr[1], root=0, op=MPI.SUM)]

        # mean increment
        if mpi.global_master and exp.n_tuples['inc'][-1] > 0:
            mean_inc /= exp.n_tuples['inc'][-1]

        # mean number of determinants
        if mpi.global_master and exp.n_tuples['inc'][-1] > 0:
            mean_ndets = np.asarray(np.rint(mean_ndets / exp.n_tuples['inc'][-1]), dtype=np.int64)

        # write restart files & save timings
        #if mpi.global_master:
        #    if calc.misc['rst']:
        #        write_file(exp.order, max_inc, 'mbe_max_inc')
        #        write_file(exp.order, min_inc, 'mbe_min_inc')
        #        write_file(exp.order, mean_inc, 'mbe_mean_inc')
        #        write_file(exp.order, max_ndets, 'mbe_max_ndets')
        #        write_file(exp.order, min_ndets, 'mbe_min_ndets')
        #        write_file(exp.order, mean_ndets, 'mbe_mean_ndets')
        #        write_file(exp.order, np.asarray(exp.n_tuples['inc'][-1]), 'mbe_idx')
        #        write_file(exp.order, hashes[-1], 'mbe_hashes')
        #        write_file(exp.order, inc[-1], 'mbe_inc')
        #    exp.time['mbe'][-1] += MPI.Wtime() - time

        # update expansion space
        exp.exp_space.append(exp.exp_space[-1])

        # write restart files
        #if mpi.global_master:
        #    if calc.misc['rst']:
        #        write_file(exp.order+1, exp.exp_space[-1], 'exp_space')

        # total property
        #tot = mean_inc * exp.n_tuples['inc'][-1]
        tot = energy

        # mpi barrier
        mpi.local_comm.Barrier()

        if mpi.global_master and pair_corr is not None and mol.debug >= 1:
            pair_corr[1] = pair_corr[1][np.argsort(np.abs(pair_corr[0]))[::-1]]
            pair_corr[0] = pair_corr[0][np.argsort(np.abs(pair_corr[0]))[::-1]]
            print('\n --------------------------------------------------------------------------')
            print(f'{"pair correlation information":^75s}')
            print(' --------------------------------------------------------------------------')
            print(' orbital tuple  |  absolute corr.  |  relative corr.  |  cumulative corr.')
            print(' --------------------------------------------------------------------------')
            for i in range(10):
                print(f'   [{pair_corr[1][i][0]:3d},{pair_corr[1][i][1]:3d}]    |' + \
                      f'    {pair_corr[0][i]:.3e}    |' + \
                      f'        {pair_corr[0][i] / pair_corr[0][0]:.2f}      |' + \
                      f'        {np.sum(pair_corr[0][:i+1]) / np.sum(pair_corr[0]):.2f}')
            print(' --------------------------------------------------------------------------\n')

        if mpi.global_master:
            return hashes_win, inc_win, tot, \
                   mean_ndets, min_ndets, max_ndets, \
                   mean_inc, min_inc, max_inc, ml_object
        else:
            return hashes_win, inc_win


def random_sample(generator: Generator[np.ndarray, None, None], len: int) -> List[np.ndarray]:
        """
        returns a random sample with length len from a generator at O(n) using
        reservoir sampling algorithm R
        """
        # fill sample
        sample = list(islice(generator, len))

        # shuffle sample
        shuffle(sample)

        for i, item in enumerate(generator, start=len+1):

            # generate random number between 0 and i
            j = randrange(i)

            # replace item with gradually decreasing probability
            if j < len:
                sample[j] = item
        
        return sample


def avail_tuples(gen: Generator[np.ndarray, None, None], order: int, min_order: int, \
                 nocc: int, ref_occ: bool, ref_virt: bool, hashes: List[np.ndarray], \
                 inc: List[np.ndarray]) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        generator function for all tuples that can be constructed from available subtuples
        """
        # loop over tuples in generator
        for tup in gen:

            # initialize all_subtups_avail
            all_subtups_avail = True

            # initialize subtup_sum
            sum_subtup = np.zeros(order - min_order, dtype=np.float64)

            # occupied and virtual subspaces of tuple
            tup_occ = tup[tup < nocc]
            tup_virt = tup[nocc <= tup]

            # get increments of subtuples
            for k in range(order-1, min_order-1, -1):

                # loop over subtuples
                for tup_sub in tuples(tup_occ, tup_virt, ref_occ, ref_virt, k):

                    # compute index
                    idx = hash_lookup(hashes[k-min_order], hash_1d(tup_sub))

                    # sum and max of subtuple increments
                    if idx is not None:

                        sum_subtup[k-min_order] += inc[k-min_order][idx]
                            
                    else:

                        all_subtups_avail = False
                        break

                if not all_subtups_avail:
                    
                    break

            if all_subtups_avail:

                yield tup, fsum(sum_subtup)

