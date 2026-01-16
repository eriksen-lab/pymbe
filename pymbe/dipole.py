#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
dipole moment expansion module
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
from typing import TYPE_CHECKING, cast

from pymbe.expansion import SingleTargetExpCls, CONV_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import RST, write_file, get_nhole, get_nexc
from pymbe.parallel import mpi_reduce, open_shared_win, mpi_bcast
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt

if TYPE_CHECKING:
    import matplotlib
    from typing import Optional, Tuple, Union, List

    from pymbe.parallel import MPICls


class DipoleExpCls(SingleTargetExpCls[np.ndarray]):
    """
    this class contains the pymbe expansion attributes for expansions of the dipole
    moment
    """

    def prop(
        self, prop_type: str, nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64)
    ) -> np.ndarray:
        """
        this function returns the final dipole moment
        """
        if len(self.mbe_tot_prop) > 0:
            tot_dipole = -self.mbe_tot_prop[-1].copy()
        else:
            tot_dipole = self._init_target_inst(0.0)
        tot_dipole -= self.base_prop
        tot_dipole -= self.ref_prop
        if prop_type in ["electronic", "total"]:
            tot_dipole -= self.hf_prop
        tot_dipole += nuc_prop

        return tot_dipole

    def plot_results(
        self, y_axis: str, nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64)
    ) -> matplotlib.figure.Figure:
        """
        this function plots the dipole moment
        """
        # array of total MBE dipole moment
        dipole = self._prop_conv(
            nuc_prop,
            (
                self.hf_prop
                if y_axis in ["electronic", "total"]
                else self._init_target_inst(0.0)
            ),
        )
        dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
        for i in range(dipole.shape[0]):
            dipole_arr[i] = np.linalg.norm(dipole[i, :])

        return results_plt(
            dipole_arr,
            self.min_order,
            self.final_order,
            "*",
            "xkcd:salmon",
            f"state {self.fci_state_root}",
            "Dipole moment (in au)",
        )

    def free_ints(self) -> None:
        """
        this function deallocates integrals in shared memory after the calculation is
        done
        """
        super().free_ints()

        # free additional integrals
        self.dipole_ints_win.Free()

        return

    def _int_wins(
        self,
        mpi: MPICls,
        hcore: Optional[np.ndarray],
        eri: Optional[np.ndarray],
        dipole_ints: Optional[np.ndarray] = None,
        **kwargs: Optional[np.ndarray],
    ):
        """
        this function creates shared memory windows for integrals on every node
        """
        super()._int_wins(mpi, hcore, eri)

        # allocate dipole integrals in shared mem
        self.dipole_ints_win: MPI.Win = MPI.Win.Allocate_shared(
            8 * 3 * self.norb**2 if mpi.local_master else 0, 8, comm=mpi.local_comm
        )

        # open dipole integrals in shared memory
        self.dipole_ints = open_shared_win(
            self.dipole_ints_win, np.float64, (3, self.norb, self.norb)
        )

        # set dipole integrals on global master
        if mpi.global_master:
            self.dipole_ints[:] = cast(np.ndarray, dipole_ints)

        # mpi_bcast dipole integrals
        if mpi.num_masters > 1 and mpi.local_master:
            self.dipole_ints[:] = mpi_bcast(mpi.master_comm, self.dipole_ints)

        # mpi barrier
        mpi.global_comm.Barrier()

        return

    def _calc_hf_prop(self, *args: np.ndarray) -> np.ndarray:
        """
        this function calculates the hartree-fock electronic dipole moment
        """
        return np.einsum("p,xpp->x", self.occup, self.dipole_ints)

    def _ref_results(self, ref_prop: np.ndarray) -> str:
        """
        this function prints reference space results for a target calculation
        """
        header = f"reference space dipole moment for root {self.fci_state_root}"
        dipole = f"(total increment = {np.linalg.norm(ref_prop):.4e})"

        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {header:^80}\n"
        string += f" RESULT: {dipole:^80}\n"
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
    ) -> np.ndarray:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # perform main calc
        dipole = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            dipole -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        dipole -= self.ref_prop

        return dipole

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
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
            conv_tol=CONV_TOL * 1.0e-04,
        )

        # init rdm1
        rdm1 = np.diag(self.occup.astype(np.float64))

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = solver.make_rdm1(
            civec[0], cas_idx.size, nelec
        )

        # compute elec_dipole
        elec_dipole = np.einsum("xij,ji->x", self.dipole_ints, rdm1)

        return elec_dipole - self.hf_prop, civec

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> np.ndarray:
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
            cas_rdm1 = ccsd.make_rdm1()
        elif method == "ccsd(t)":
            if spin_cas == 0:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                cas_rdm1 = ccsd_t_rdm.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                cas_rdm1 = uccsd_t_rdm.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )

        if spin_cas != 0:
            cas_rdm1 = cas_rdm1[0] + cas_rdm1[1]

        # init rdm1
        rdm1 = np.diag(self.occup.astype(np.float64))

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

        # compute elec_dipole
        elec_dipole = np.einsum("xij,ji->x", self.dipole_ints, rdm1)

        return elec_dipole - self.hf_prop

    @staticmethod
    def _write_target_file(prop: np.ndarray, string: str, order: int) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(prop, string, order=order)

    @staticmethod
    def _read_target_file(file: str) -> np.ndarray:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file))

    def _init_target_inst(self, value: float, *args: int) -> np.ndarray:
        """
        this function initializes an instance of the target type
        """
        return np.full(3, value, dtype=np.float64)

    def _zero_target_arr(self, length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return np.zeros((length, 3), dtype=np.float64)

    @staticmethod
    def _mpi_reduce_target(
        comm: MPI.Comm, values: np.ndarray, op: MPI.Op
    ) -> np.ndarray:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return mpi_reduce(comm, values, root=0, op=op)

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Intracomm, *args: int
    ) -> Optional[MPI.Win]:
        """
        this function allocates a shared increment window
        """
        return (
            MPI.Win.Allocate_shared(8 * size * 3 if allocate else 0, 8, comm=comm)
            if size > 0
            else None
        )

    def _open_shared_inc(
        self, window: Optional[MPI.Win], n_incs: int, *args: int
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """
        return open_shared_win(window, np.float64, (n_incs, 3))

    def _add_screen(
        self,
        inc_tup: np.ndarray,
        order: int,
        tup: np.ndarray,
        *args: Optional[List[np.ndarray]],
    ) -> None:
        """
        this function modifies the screening array
        """
        self.screen[order - 1]["max"][tup] = np.maximum(
            self.screen[order - 1]["max"][tup], np.max(np.abs(inc_tup))
        )
        self.screen[order - 1]["sum_abs"][tup] += np.sum(np.abs(inc_tup))

    @staticmethod
    def _total_inc(inc: List[np.ndarray], mean_inc: np.ndarray) -> np.ndarray:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc.copy()

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: np.ndarray,
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
            f"      increment for root {self.fci_state_root:d} = ({inc_tup[0]:.4e}, "
            f"{inc_tup[1]:.4e}, {inc_tup[2]:.4e})\n"
        )

        return string

    def _adaptive_screen(self, inc: List[List[np.ndarray]]):
        """
        this function wraps the adaptive screening function
        """
        raise NotImplementedError

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for a dipole or transition dipole
        calculation
        """
        # calculate total inc
        if order == self.min_order:
            tot_inc = np.linalg.norm(self.mbe_tot_prop[order - self.min_order])
        else:
            tot_inc = np.linalg.norm(
                self.mbe_tot_prop[order - self.min_order]
            ) - np.linalg.norm(self.mbe_tot_prop[order - self.min_order - 1])

        # set header
        if self.target == "dipole":
            header = (
                f"dipole moment for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )
        elif self.target == "trans":
            header = (
                f"transition dipole moment for excitation 0 -> {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{header:^81}\n"
        string += DIVIDER_OUTPUT + "\n"

        if not self.closed_form:
            # set string
            string += DIVIDER_OUTPUT

            # loop over x, y, and z
            comp = ("x-component", "y-component", "z-component")
            for k in range(3):
                string += f"\n RESULT-{order:d}:{comp[k]:^81}\n"
                string += DIVIDER_OUTPUT + "\n"
                string += (
                    f" RESULT-{order:d}:      mean increment     |      "
                    "min. abs. increment     |     max. abs. increment\n"
                )
                string += DIVIDER_OUTPUT + "\n"
                string += (
                    f" RESULT-{order:d}:     "
                    f"{self.mean_inc[order - self.min_order][k]:>13.4e}       |        "
                    f"{self.min_inc[order - self.min_order][k]:>13.4e}         |       "
                    f"{self.max_inc[order - self.min_order][k]:>13.4e}\n"
                )
                if k < 2:
                    string += DIVIDER_OUTPUT

            string += DIVIDER_OUTPUT + "\n"

        # set string
        string += FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT

        return string

    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the norm of the hf, base and total electronic dipole moment
        """
        hf_prop = np.linalg.norm(self.hf_prop)
        base_prop = np.linalg.norm(self.hf_prop + self.base_prop)
        mbe_tot_prop = np.linalg.norm(self._prop_conv(hf_prop=self.hf_prop)[-1, :])

        return hf_prop, base_prop, mbe_tot_prop

    def _results_prt(self) -> str:
        """
        this function returns the dipole moments table
        """
        string: str = DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"MBE electronic dipole moment (root = {self.fci_state_root})".center(87)
            + "\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'elec. dipole components (x,y,z)':^43}{'|':1}"
            f"{'elec. dipole moment':^21}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        tot_ref_dipole: np.ndarray = -(self.hf_prop + self.base_prop + self.ref_prop)
        string += (
            f"{'':3}{'ref':^14s}{'|':1}{tot_ref_dipole[0]:>13.6f}"
            f"{tot_ref_dipole[1]:>13.6f}{tot_ref_dipole[2]:>13.6f}{'':4}{'|':1}"
            f"{np.linalg.norm(tot_ref_dipole):>14.6f}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        dipole = self._prop_conv(hf_prop=self.hf_prop)
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}{dipole[i, 0]:>13.6f}"
                f"{dipole[i, 1]:>13.6f}{dipole[i, 2]:>13.6f}{'':4}{'|':1}"
                f"{np.linalg.norm(dipole[i, :]):>14.6f}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:83] + "\n"

        return string

    def _prop_conv(
        self,
        nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64),
        hf_prop: np.ndarray = np.zeros(3, dtype=np.float64),
    ) -> np.ndarray:
        """
        this function returns the total dipole moment
        """
        tot_dipole = -np.array(self.mbe_tot_prop)
        tot_dipole -= self.base_prop
        tot_dipole -= self.ref_prop
        tot_dipole -= hf_prop
        tot_dipole += nuc_prop

        return tot_dipole
