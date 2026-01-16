#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy expansion module
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
from pyscf import ao2mo
from typing import TYPE_CHECKING

from pymbe.logger import logger
from pymbe.expansion import SingleTargetExpCls, CONV_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    write_file,
    idx_tril,
    get_nhole,
    get_nexc,
    add_inc_stats,
    adaptive_screen_dict,
)
from pymbe.parallel import mpi_reduce, open_shared_win
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt
from pymbe.interface import mbecc_interface
from pymbe.screen import adaptive_screen

if TYPE_CHECKING:
    import matplotlib
    from typing import Tuple, Union, Dict, List, Optional


class EnergyExpCls(SingleTargetExpCls[float]):
    """
    this class contains the pymbe expansion attributes for expansions of the electronic
    energy
    """

    def prop(self, prop_type: str, nuc_prop: float = 0.0) -> float:
        """
        this function returns the final energy
        """
        if len(self.mbe_tot_prop) > 0:
            tot_energy = self.mbe_tot_prop[-1]
        else:
            tot_energy = self._init_target_inst(0.0)
        tot_energy += self.ref_prop
        tot_energy += self.base_prop
        if prop_type in ["electronic", "total"]:
            tot_energy += self.hf_prop
        tot_energy += nuc_prop

        return tot_energy

    def plot_results(
        self, y_axis: str, nuc_prop: float = 0.0
    ) -> matplotlib.figure.Figure:
        """
        this function plots the energy
        """
        return results_plt(
            self._prop_conv(
                nuc_prop,
                (
                    self.hf_prop
                    if y_axis in ["electronic", "total"]
                    else self._init_target_inst(0.0)
                ),
            ),
            self.min_order,
            self.final_order,
            "x",
            "xkcd:kelly green",
            f"state {self.fci_state_root}",
            "Energy (in au)",
        )

    def _calc_hf_prop(
        self, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
    ) -> float:
        """
        this function calculates the hartree-fock electronic energy
        """
        # add one-electron integrals
        hf_energy = np.sum(self.occup * np.diag(hcore))

        # set closed- and open-shell indices
        cs_idx = np.where(self.occup == 2)[0]
        os_idx = np.where(self.occup == 1)[0]

        # add closed-shell and coupling electron repulsion terms
        hf_energy += np.trace((np.sum(vhf, axis=0))[cs_idx.reshape(-1, 1), cs_idx])

        # check if system is open-shell
        if self.spin > 0:
            # get indices for eris that only include open-shell orbitals
            os_eri_idx = idx_tril(os_idx)

            # retrieve eris of open-shell orbitals and unpack these
            os_eri = ao2mo.restore(
                1, eri[os_eri_idx.reshape(-1, 1), os_eri_idx], os_idx.size
            )

            # add open-shell electron repulsion terms
            hf_energy += 0.5 * (
                np.einsum("pqrr->", os_eri) - np.einsum("pqrp->", os_eri)
            )

        return hf_energy

    def _ref_results(self, ref_prop: float) -> str:
        """
        this function prints reference space results for a target calculation
        """
        header = (
            f"reference space energy for root {self.fci_state_root} "
            + f"(total increment = {ref_prop:.4e})"
        )

        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {header:^80}\n"
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
    ) -> float:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # perform main calc
        energy = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            energy -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        energy -= self.ref_prop

        return energy

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool,
    ) -> Tuple[float, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # run fci calculation
        energy, civec, _ = self._fci_driver(
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

        return energy[0] - self.hf_prop, civec

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> float:
        """
        this function returns the results of a cc calculation
        """
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"cascc wrong spin in space: {cas_idx}")

        if self.cc_backend == "pyscf":
            # run ccsd calculation
            e_cc, ccsd, _ = self._ccsd_driver_pyscf(
                h1e, h2e, core_idx, cas_idx, spin_cas
            )

            # calculate (t) correction
            if method == "ccsd(t)":
                # number of holes in cas space
                nhole = get_nhole(nelec, cas_idx)

                # nexc
                nexc = get_nexc(nelec, nhole)

                # ensure that more than two excitations are possible
                if nexc > 2:
                    e_cc += ccsd.ccsd_t()

        elif self.cc_backend in ["ecc", "ncc"]:
            # calculate cc energy
            e_cc, success = mbecc_interface(
                method,
                self.cc_backend,
                self.orb_type,
                self.point_group,
                self.orbsym[cas_idx],
                h1e,
                h2e,
                nelec,
                higher_amp_extrap,
                self.verbose,
            )

            # convergence check
            if not success == 1:
                raise RuntimeError(
                    f"MBECC error: no convergence, core_idx = {core_idx}, "
                    f"cas_idx = {cas_idx}"
                )

        return e_cc

    @staticmethod
    def _write_target_file(prop: float, string: str, order: int) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(np.array(prop, dtype=np.float64), string, order=order)

    @staticmethod
    def _read_target_file(file: str) -> float:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file)).item()

    def _init_target_inst(self, value: float, *args: int) -> float:
        """
        this function initializes an instance of the target type
        """
        return value

    def _init_screen(self) -> Dict[str, np.ndarray]:
        """
        this function initializes the screening arrays
        """
        return {
            "sum_abs": np.zeros(self.norb, dtype=np.float64),
            "sum": np.zeros(self.norb, dtype=np.float64),
            "max": np.zeros(self.norb, dtype=np.float64),
        }

    def _zero_target_arr(self, length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return np.zeros(length, dtype=np.float64)

    @staticmethod
    def _mpi_reduce_target(comm: MPI.Comm, values: float, op: MPI.Op) -> float:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return mpi_reduce(
            comm, np.array(values, dtype=np.float64), root=0, op=op
        ).item()

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Intracomm, *args: int
    ) -> Optional[MPI.Win]:
        """
        this function allocates a shared increment window
        """
        return (
            MPI.Win.Allocate_shared(8 * size if allocate else 0, 8, comm=comm)
            if size > 0
            else None
        )

    def _open_shared_inc(
        self, window: Optional[MPI.Win], n_incs: int, *args: int
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """
        return open_shared_win(window, np.float64, (n_incs,))

    def _add_screen(
        self,
        inc_tup: float,
        order: int,
        tup: np.ndarray,
        tup_clusters: Optional[List[np.ndarray]],
    ) -> None:
        """
        this function modifies the screening array
        """
        # get absolute increment
        abs_inc_tup = np.abs(inc_tup)

        # get screening values for static screening
        self.screen[order - 1]["max"][tup] = np.maximum(
            self.screen[order - 1]["max"][tup], abs_inc_tup
        )
        self.screen[order - 1]["sum_abs"][tup] += abs_inc_tup
        self.screen[order - 1]["sum"][tup] += inc_tup

        # get screening values for adaptive screening
        if self.screen_type == "adaptive":
            # add values for increment
            if abs_inc_tup > 0.0:
                self.adaptive_screen = add_inc_stats(
                    abs_inc_tup,
                    tup,
                    tup_clusters,
                    self.adaptive_screen,
                    self.nocc,
                    self.order,
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                )

    @staticmethod
    def _total_inc(inc: List[np.ndarray], mean_inc: float) -> float:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: float,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        string += f"      increment for root {self.fci_state_root:d} = {inc_tup:.4e}\n"

        return string

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an energy or excitation energy
        calculation
        """
        # calculate total inc
        if order == self.min_order:
            tot_inc = self.mbe_tot_prop[order - self.min_order]
        else:
            tot_inc = (
                self.mbe_tot_prop[order - self.min_order]
                - self.mbe_tot_prop[order - self.min_order - 1]
            )

        # set header
        if self.target == "energy":
            header = (
                f"energy for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )
        elif self.target == "excitation":
            header = (
                f"excitation energy for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{header:^81}\n"
        string += DIVIDER_OUTPUT + "\n"

        if not self.closed_form:
            # set string
            string += DIVIDER_OUTPUT + "\n"
            string += (
                f" RESULT-{order:d}:      mean increment     |      "
                "min. abs. increment     |     max. abs. increment\n"
            )
            string += DIVIDER_OUTPUT + "\n"
            string += (
                f" RESULT-{order:d}:     "
                f"{self.mean_inc[order - self.min_order]:>13.4e}       |        "
                f"{self.min_inc[order - self.min_order]:>13.4e}         |       "
                f"{self.max_inc[order - self.min_order]:>13.4e}\n"
            )
            string += DIVIDER_OUTPUT + "\n"

        # set string
        string += FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT

        return string

    def _adaptive_screen(self, incs: List[List[np.ndarray]]):
        """
        this function wraps the adaptive screening function
        """
        return adaptive_screen(
            self.screen_tot_prop,
            self.mbe_tot_error[-1],
            self.screen_thres,
            adaptive_screen_dict(self.adaptive_screen, self.norb),
            self.exp_clusters[-1],
            self.exp_space[-1],
            self.exp_single_orbs,
            incs,
            self.nocc,
            self.ref_nelec,
            self.ref_nhole,
            self.vanish_exc,
            self.order,
            CONV_TOL,
        )

    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the hf, base and total electronic energy
        """
        return (
            self.hf_prop,
            self.hf_prop + self.base_prop,
            self._prop_conv(hf_prop=self.hf_prop)[-1],
        )

    def _results_prt(self) -> str:
        """
        this function returns the energies table
        """
        string: str = DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"MBE-{self.method.upper()} electronic energy (root = {self.fci_state_root})"
        ).center(73) + "\n"

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'electronic energy':^22}{'|':1}"
            f"{'correlation energy':^26}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'ref':^14s}{'|':1}"
            f"{(self.hf_prop + self.ref_prop):^22.6f}{'|':1}"
            f"{self.ref_prop:>19.5e}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        energy = self._prop_conv(hf_prop=self.hf_prop)
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}"
                f"{energy[i]:^22.6f}{'|':1}"
                f"{(energy[i] - self.hf_prop):>19.5e}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:67] + "\n"

        return string

    def _prop_conv(self, nuc_prop: float = 0.0, hf_prop: float = 0.0) -> np.ndarray:
        """
        this function returns the total energy
        """
        tot_energy = np.array(self.mbe_tot_prop)
        tot_energy += self.base_prop
        tot_energy += self.ref_prop
        tot_energy += hf_prop
        tot_energy += nuc_prop

        return tot_energy
