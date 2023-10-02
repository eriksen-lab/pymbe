#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
1- and 2-particle reduced density matrix expansion module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import numpy as np
from mpi4py import MPI
from pyscf.cc import (
    ccsd_t_lambda_slow as ccsd_t_lambda,
    ccsd_t_rdm_slow as ccsd_t_rdm,
    uccsd_t_lambda,
    uccsd_t_rdm,
)
from typing import TYPE_CHECKING, TypedDict, Tuple, List

from pymbe.expansion import ExpCls, StateIntType, StateArrayType
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    RDMCls,
    packedRDMCls,
    get_nelec,
    tuples,
    hash_1d,
    hash_lookup,
    get_occup,
    get_nhole,
    get_nexc,
    core_cas,
    write_file_mult,
)
from pymbe.parallel import open_shared_win, mpi_reduce, mpi_allreduce, mpi_bcast

if TYPE_CHECKING:
    import matplotlib
    from typing import Optional, Union, Dict


class RDMExpCls(
    ExpCls[
        RDMCls,
        Tuple[np.ndarray, np.ndarray],
        packedRDMCls,
        Tuple[MPI.Win, MPI.Win],
        StateIntType,
        StateArrayType,
    ]
):
    """
    this class contains the pymbe expansion attributes
    """

    def __del__(self) -> None:
        """
        finalizes expansion attributes
        """
        # ensure the class attributes of packedRDMCls are reset
        packedRDMCls.reset()

    def prop(self, prop_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the final 1- and 2-particle reduced density matrices
        """
        tot_rdm12 = self.mbe_tot_prop[-1].copy()
        tot_rdm12[self.ref_space] += self.ref_prop
        tot_rdm12 += self.base_prop
        if prop_type in ["electronic", "total"]:
            core_idx, cas_idx = core_cas(self.nocc, self.ref_space, self.exp_space[0])
            tot_rdm12[core_idx] += self.hf_prop[core_idx]
            tot_rdm12[cas_idx] += self.hf_prop[cas_idx]

        return tot_rdm12.rdm1, tot_rdm12.rdm2

    def plot_results(self, *args: str) -> matplotlib.figure.Figure:
        """
        this function plots the 1- and 2-particle reduced density matrices
        """
        raise NotImplementedError

    @staticmethod
    def _convert_to_target(prop: Tuple[np.ndarray, np.ndarray]) -> RDMCls:
        """
        this function converts the input target type into the used target type
        """
        return RDMCls(*prop)

    def _calc_hf_prop(self, *args: np.ndarray) -> RDMCls:
        """
        this function calculates the hartree-fock reduced density matrices
        """
        rdm1 = np.zeros(2 * (self.norb,), dtype=np.float64)
        np.einsum("ii->i", rdm1)[...] += self.occup

        rdm2 = np.zeros(4 * (self.norb,), dtype=np.float64)
        occup_a = self.occup.copy()
        occup_a[occup_a > 0.0] = 1.0
        occup_b = self.occup - occup_a
        # d_ppqq = k_pa*k_qa + k_pb*k_qb + k_pa*k_qb + k_pb*k_qa = k_p*k_q
        np.einsum("iijj->ij", rdm2)[...] += np.einsum("i,j", self.occup, self.occup)
        # d_pqqp = - (k_pa*k_qa + k_pb*k_qb)
        np.einsum("ijji->ij", rdm2)[...] -= np.einsum(
            "i,j", occup_a, occup_a
        ) + np.einsum("i,j", occup_b, occup_b)

        return RDMCls(rdm1, rdm2)

    def _ref_results(self, ref_prop: RDMCls) -> str:
        """
        this function prints reference space results for a target calculation
        """
        rdm1 = (
            f"reference space 1-rdm for root {self.fci_state_root} "
            + f"(total increment norm = {np.linalg.norm(ref_prop.rdm1):.4e})"
        )
        rdm2 = (
            f"reference space 2-rdm for root {self.fci_state_root} "
            f"(total increment norm = {np.linalg.norm(ref_prop.rdm2):.4e})"
        )

        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT:{rdm1:^81}\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT:{rdm2:^81}\n"
        string += DIVIDER_OUTPUT

        return string

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> RDMCls:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # perform main calc
        rdm12 = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            rdm12 -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        ind = np.where(np.in1d(cas_idx, self.ref_space))[0]

        rdm12[ind] -= self.ref_prop

        return rdm12

    def _sum(
        self,
        inc: List[packedRDMCls],
        hashes: List[np.ndarray],
        tup: np.ndarray,
    ) -> RDMCls:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """
        # init res
        res = RDMCls(
            np.zeros(
                [self.ref_space.size + self.order, self.ref_space.size + self.order],
                dtype=np.float64,
            ),
            np.zeros(
                [
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                ],
                dtype=np.float64,
            ),
        )

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < self.nocc]
        tup_virt = tup[self.nocc <= tup]

        # rank of reference space and occupied and virtual tuple orbitals
        rank = np.argsort(np.argsort(np.concatenate((self.ref_space, tup))))
        ind_ref = rank[: self.ref_space.size]
        ind_tup_occ = rank[self.ref_space.size : self.ref_space.size + tup_occ.size]
        ind_tup_virt = rank[self.ref_space.size + tup_occ.size :]

        # compute contributions from lower-order increments
        for k in range(self.order - 1, self.min_order - 1, -1):
            # rank of all orbitals in casci space
            ind_casci = np.empty(self.ref_space.size + k, dtype=np.int64)

            # loop over subtuples
            for tup_sub, ind_sub in zip(
                tuples(
                    tup_occ,
                    tup_virt,
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    k,
                ),
                tuples(
                    ind_tup_occ,
                    ind_tup_virt,
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    k,
                ),
            ):
                # compute index
                idx = hash_lookup(hashes[k - self.min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    # add rank of reference space orbitals
                    ind_casci[: self.ref_space.size] = ind_ref

                    # add rank of subtuple orbitals
                    ind_casci[self.ref_space.size :] = ind_sub

                    # sort indices for faster assignment
                    ind_casci.sort()

                    # add subtuple rdms
                    res[ind_casci] += inc[k - self.min_order][idx]

        return res

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> RDMCls:
        """
        this function returns the results of a cc calculation
        """
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"cascc wrong spin in space: {cas_idx}")

        # number of holes in cas space
        nhole = get_nhole(nelec, cas_idx)

        # number of possible excitations in cas space
        nexc = get_nexc(nelec, nhole)

        # run ccsd calculation
        _, ccsd, eris = self._ccsd_driver_pyscf(
            h1e, h2e, core_idx, cas_idx, spin_cas, converge_amps=True
        )

        # rdms
        if method == "ccsd" or nexc <= 2:
            ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
            rdm1 = ccsd.make_rdm1()
            rdm2 = ccsd.make_rdm2()
        elif method == "ccsd(t)":
            if spin_cas == 0:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                rdm2 = ccsd_t_rdm.make_rdm2(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                rdm2 = uccsd_t_rdm.make_rdm2(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

        if spin_cas == 0:
            rdm12 = RDMCls(rdm1, rdm2)
        else:
            rdm12 = RDMCls(rdm1[0] + rdm1[1], rdm2[0] + rdm2[1] + rdm2[2] + rdm2[3])
        return rdm12 - self.hf_prop[cas_idx]

    @staticmethod
    def _write_target_file(prop: RDMCls, string: str, order: Optional[int]) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file_mult({"rdm1": prop.rdm1, "rdm2": prop.rdm2}, string, order)

    @staticmethod
    def _read_target_file(file: str) -> RDMCls:
        """
        this function reads files of attributes with the target type
        """
        rdm_dict = np.load(os.path.join(RST, file))
        return RDMCls(rdm_dict["rdm1"], rdm_dict["rdm2"])

    def _init_target_inst(self, value: float, norb: int) -> RDMCls:
        """
        this function initializes an instance of the target type
        """
        return RDMCls(
            np.full(2 * (norb,), value, dtype=np.float64),
            np.full(4 * (norb,), value, dtype=np.float64),
        )

    @staticmethod
    def _mpi_reduce_target(comm: MPI.Comm, values: RDMCls, op: MPI.Op) -> RDMCls:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return RDMCls(
            mpi_reduce(comm, values.rdm1, root=0, op=op),
            mpi_reduce(comm, values.rdm2, root=0, op=op),
        )

    @staticmethod
    def _write_inc_file(inc: packedRDMCls, order: Optional[int]) -> None:
        """
        this function defines how to write increment restart files
        """
        if order is None:
            np.savez(os.path.join(RST, "mbe_inc"), rdm1=inc.rdm1, rdm2=inc.rdm2)
        else:
            np.savez(
                os.path.join(RST, f"mbe_inc_{order}"), rdm1=inc.rdm1, rdm2=inc.rdm2
            )

    @staticmethod
    def _read_inc_file(file: str) -> packedRDMCls:
        """
        this function defines reads the increment restart files
        """
        rdm_dict = np.load(os.path.join(RST, file))

        return packedRDMCls(rdm_dict["rdm1"], rdm_dict["rdm2"])

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> Tuple[MPI.Win, MPI.Win]:
        """
        this function allocates a shared increment window
        """
        # get current length of incs array
        idx = len(self.incs)

        # generate packing and unpacking indices if these have not been generated yet
        if len(packedRDMCls.rdm1_size) == idx:
            packedRDMCls.get_pack_idx(self.ref_space.size + self.min_order + idx)

        return (
            MPI.Win.Allocate_shared(
                8 * size * packedRDMCls.rdm1_size[idx] if allocate else 0,
                8,
                comm=comm,  # type: ignore
            ),
            MPI.Win.Allocate_shared(
                8 * size * packedRDMCls.rdm2_size[idx] if allocate else 0,
                8,
                comm=comm,  # type: ignore
            ),
        )

    def _open_shared_inc(
        self, window: Tuple[MPI.Win, MPI.Win], n_tuples: int, idx: int
    ) -> packedRDMCls:
        """
        this function opens a shared increment window
        """
        # open shared windows
        rdm1 = open_shared_win(
            window[0], np.float64, (n_tuples, packedRDMCls.rdm1_size[idx])
        )
        rdm2 = open_shared_win(
            window[1], np.float64, (n_tuples, packedRDMCls.rdm2_size[idx])
        )

        return packedRDMCls(rdm1, rdm2, idx)

    def _init_inc_arr_from_lst(self, inc_lst: List[RDMCls]) -> packedRDMCls:
        """
        this function creates an increment array from a list of increments
        """
        # initialize arrays
        rdm1 = np.empty((len(inc_lst), packedRDMCls.rdm1_size[-1]), dtype=np.float64)
        rdm2 = np.empty((len(inc_lst), packedRDMCls.rdm2_size[-1]), dtype=np.float64)

        # fill arrays
        for i, inc in enumerate(inc_lst):
            rdm1[i] = inc.rdm1[packedRDMCls.pack_rdm1[-1]]
            rdm2[i] = inc.rdm2[packedRDMCls.pack_rdm2[-1]]

        return packedRDMCls(rdm1, rdm2)

    @staticmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: packedRDMCls) -> packedRDMCls:
        """
        this function bcasts the increments
        """
        return packedRDMCls(
            mpi_bcast(comm, inc.rdm1),
            mpi_bcast(comm, inc.rdm2),
        )

    @staticmethod
    def _mpi_reduce_inc(comm: MPI.Comm, inc: packedRDMCls, op: MPI.Op) -> packedRDMCls:
        """
        this function performs a MPI reduce operation on the increments
        """
        return packedRDMCls(
            mpi_reduce(comm, inc.rdm1, root=0, op=op),
            mpi_reduce(comm, inc.rdm2, root=0, op=op),
        )

    @staticmethod
    def _mpi_allreduce_inc(
        comm: MPI.Comm, inc: packedRDMCls, op: MPI.Op
    ) -> packedRDMCls:
        """
        this function performs a MPI allreduce operation on the increments
        """
        return packedRDMCls(
            mpi_allreduce(comm, inc.rdm1, op=op),
            mpi_allreduce(comm, inc.rdm2, op=op),
        )

    @staticmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: packedRDMCls, recv_inc: Optional[packedRDMCls]
    ) -> None:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # size of arrays on every rank
        counts_dict = {
            "rdm1": np.array(comm.gather(send_inc.rdm1.size)),
            "rdm2": np.array(comm.gather(send_inc.rdm2.size)),
        }

        # receiving arrays
        recv_inc_dict: Dict[str, Optional[np.ndarray]] = {}
        if recv_inc is not None:
            recv_inc_dict["rdm1"] = recv_inc.rdm1
            recv_inc_dict["rdm2"] = recv_inc.rdm2
        else:
            recv_inc_dict["rdm1"] = recv_inc_dict["rdm2"] = None

        comm.Gatherv(
            send_inc.rdm1.ravel(), (recv_inc_dict["rdm1"], counts_dict["rdm1"]), root=0
        )
        comm.Gatherv(
            send_inc.rdm2.ravel(), (recv_inc_dict["rdm2"], counts_dict["rdm2"]), root=0
        )

    @staticmethod
    def _free_inc(inc_win: Tuple[MPI.Win, MPI.Win]) -> None:
        """
        this function frees the supplied increment windows
        """
        inc_win[0].Free()
        inc_win[1].Free()

    @staticmethod
    def _add_screen(
        inc_tup: RDMCls, screen: np.ndarray, tup: np.ndarray, screen_func: str
    ) -> np.ndarray:
        """
        this function modifies the screening array
        """
        if screen_func == "max":
            return np.maximum(screen[tup], np.max(np.abs(inc_tup.rdm1)))
        elif screen_func == "sum_abs":
            return screen[tup] + np.sum(np.abs(inc_tup.rdm1))
        else:
            raise ValueError

    def _update_inc_stats(
        self,
        inc_tup: RDMCls,
        min_inc: RDMCls,
        mean_inc: RDMCls,
        max_inc: RDMCls,
        cas_idx: np.ndarray,
        neqv_tups: int,
    ) -> Tuple[RDMCls, RDMCls, RDMCls]:
        """
        this function updates the increment statistics
        """
        # add to total rdm
        mean_inc[cas_idx] += inc_tup

        return min_inc, mean_inc, max_inc

    @staticmethod
    def _total_inc(inc: packedRDMCls, mean_inc: RDMCls) -> RDMCls:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc.copy()

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an rdm12 calculation
        """
        # calculate total inc
        if order == self.min_order:
            tot_inc = self.mbe_tot_prop[order - self.min_order]
        else:
            tot_inc = (
                self.mbe_tot_prop[order - self.min_order]
                - self.mbe_tot_prop[order - self.min_order - 1]
            )

        # set headers
        rdm1 = (
            f"1-rdm for root {self.fci_state_root} "
            + f"(total increment norm = {np.linalg.norm(tot_inc.rdm1):.4e})"
        )
        rdm2 = (
            f"2-rdm for root {self.fci_state_root} "
            f"(total increment norm = {np.linalg.norm(tot_inc.rdm2):.4e})"
        )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{rdm1:^81}\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{rdm2:^81}\n"

        # set string
        string += DIVIDER_OUTPUT + "\n"
        string += FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT

        return string

    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the hf, base and total 1- and 2-particle reduced density
        matrices
        """
        raise NotImplementedError

    def _results_prt(self) -> str:
        """
        this function returns the 1- and 2-particle reduced density matrices table
        """
        raise NotImplementedError


class ssRDMExpCls(RDMExpCls[int, np.ndarray]):
    """
    this class contains the pymbe expansion attributes
    """

    def _state_occup(self) -> None:
        """
        this function initializes certain state attributes for a single state
        """
        self.nocc = np.max(self.nelec)
        self.spin = abs(self.nelec[0] - self.nelec[1])
        self.occup = get_occup(self.norb, self.nelec)

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool = True,
    ) -> Tuple[RDMCls, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # run fci calculation
        _, civec, solver = self._fci_driver(
            e_core,
            h1e,
            h2e,
            cas_idx,
            nelec,
            spin_cas,
            self.fci_state_sym,
            [self.fci_state_root],
            ref_guess,
        )

        return (
            RDMCls(*solver.make_rdm12(civec[0], cas_idx.size, nelec))
            - self.hf_prop[cas_idx],
            civec,
        )

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: RDMCls,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        string += (
            f"      rdm1 increment for root {self.fci_state_root:d} = "
            + np.array2string(inc_tup.rdm1, max_line_width=59, precision=4)
            + "\n"
        )
        string += (
            f"      rdm2 increment for root {self.fci_state_root:d} = "
            + np.array2string(inc_tup.rdm2, max_line_width=59, precision=4)
            + "\n"
        )

        return string


class saRDMExpCls(RDMExpCls[List[int], List[np.ndarray]]):
    """
    this class contains the pymbe expansion attributes
    """

    def _state_occup(self) -> None:
        """
        this function initializes certain state attributes for multiple states
        """
        self.nocc = max([np.max(self.nelec[0])])
        self.spin = [abs(state[0] - state[1]) for state in self.nelec]
        self.occup = get_occup(self.norb, self.nelec[0])

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        _nelec: np.ndarray,
        ref_guess: bool = True,
    ) -> Tuple[RDMCls, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # initialize state-averaged RDMs
        sa_rdm12 = RDMCls(
            np.zeros(2 * (cas_idx.size,), dtype=np.float64),
            np.zeros(4 * (cas_idx.size,), dtype=np.float64),
        )

        # get the total number of states
        states = [
            {"spin": spin, "sym": sym, "root": root}
            for spin, sym, root in zip(
                self.spin, self.fci_state_sym, self.fci_state_root
            )
        ]

        # define dictionary for solver settings
        class SolverDict(TypedDict):
            spin: int
            nelec: np.ndarray
            sym: int
            states: List[int]

        # get unique solvers
        solvers: List[SolverDict] = []

        # loop over states
        for n, state in enumerate(states):
            # nelec
            occup = np.zeros(self.norb, dtype=np.int64)
            occup[: np.amin(self.nelec[n])] = 2
            occup[np.amin(self.nelec[n]) : np.amax(self.nelec[n])] = 1
            nelec = get_nelec(occup, cas_idx)

            # spin
            spin_cas = abs(nelec[0] - nelec[1])
            if spin_cas != state["spin"]:
                raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

            # loop over solver settings
            for solver_info in solvers:
                # determine if state is already described by solver
                if (
                    state["spin"] == solver_info["spin"]
                    and state["sym"] == solver_info["sym"]
                ):
                    # add state to solver
                    solver_info["states"].append(n)
                    break

            # no solver describes state
            else:
                # add new solver
                solvers.append(
                    {
                        "spin": state["spin"],
                        "nelec": nelec,
                        "sym": state["sym"],
                        "states": [n],
                    }
                )

        # loop over solvers
        for solver_info in solvers:
            # get roots for this solver
            roots = [states[state]["root"] for state in solver_info["states"]]

            # run fci calculation
            _, civec, solver = self._fci_driver(
                e_core,
                h1e,
                h2e,
                cas_idx,
                solver_info["nelec"],
                solver_info["spin"],
                solver_info["sym"],
                roots,
                ref_guess,
            )

            # calculate state-averaged 1- and 2-RDMs
            for root, state_idx in zip(roots, solver_info["states"]):
                sa_rdm12 += self.fci_state_weights[state_idx] * (
                    RDMCls(
                        *solver.make_rdm12(
                            civec[root], cas_idx.size, solver_info["nelec"]
                        )
                    )
                    - self.hf_prop[cas_idx]
                )

        return sa_rdm12, civec

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: RDMCls,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        string += (
            f"      rdm1 increment for averaged states = "
            + np.array2string(inc_tup.rdm1, max_line_width=59, precision=4)
            + "\n"
        )
        string += (
            f"      rdm2 increment for averaged states = "
            + np.array2string(inc_tup.rdm2, max_line_width=59, precision=4)
            + "\n"
        )

        return string
