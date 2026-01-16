#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
generalized Fock matrix expansion module
(occ-general and active-general)
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import sys
import numpy as np
from mpi4py import MPI
from pyscf import ao2mo
from pyscf.cc import (
    ccsd_t_lambda_slow as ccsd_t_lambda,
    ccsd_t_rdm_slow as ccsd_t_rdm,
    uccsd_t_lambda,
    uccsd_t_rdm,
)
from typing import TYPE_CHECKING, cast, Tuple, List

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from pymbe.logger import logger
from pymbe.expansion import ExpCls, StateIntType, StateArrayType
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    GenFockCls,
    packedGenFockCls,
    RDMCls,
    get_nelec,
    tuples_idx_virt_idx_with_nocc,
    hash_1d,
    hash_lookup,
    get_occup,
    get_nhole,
    get_nexc,
    idx_tril,
    write_file_mult,
)
from pymbe.parallel import mpi_reduce, mpi_allreduce, mpi_bcast, open_shared_win

if TYPE_CHECKING:
    import matplotlib
    from typing import Optional, Union, Dict

    from pymbe.parallel import MPICls


class GenFockExpCls(
    ExpCls[
        GenFockCls,
        Tuple[float, np.ndarray, np.ndarray],
        packedGenFockCls,
        Tuple[MPI.Win, MPI.Win, MPI.Win],
        StateIntType,
        StateArrayType,
    ]
):
    """
    this class contains the pymbe expansion attributes for the generalized Fock matrix
    elements
    """

    def __del__(self) -> None:
        """
        finalizes expansion attributes
        """
        # ensure the class attributes of packedGenFockCls are reset
        packedGenFockCls.reset()

    def prop(self, prop_type: str) -> Tuple[float, np.ndarray]:
        """
        this function returns the final generalized Fock matrix elements
        """
        if len(self.mbe_tot_prop) > 0:
            tot_targets = self.mbe_tot_prop[-1].copy()
        else:
            tot_targets = self._init_target_inst(0.0, self.norb, self.nocc)
        tot_targets += self.base_prop
        tot_targets[
            self.ref_space,
            np.concatenate((np.arange(self.nocc), self.ref_virt)),
        ] += self.ref_prop
        if prop_type in ["electronic", "total"]:
            tot_targets += self.hf_prop

        # initialize occupied-general and occupied-active blocks of generalized Fock
        # matrix
        tot_gen_fock = np.empty(
            (self.full_nocc + self.norb, self.full_norb), dtype=np.float64
        )

        # add inactive Fock matrix elements
        inact_fock_pi = self.inact_fock[:, : self.full_nocc]

        # calculate active Fock matrix elements
        act_fock_pi = np.einsum(
            "uv,piuv->pi",
            tot_targets.rdm1,
            self.eri_goaa - 0.5 * self.eri_gaao.transpose(0, 3, 2, 1),
        )

        # calculate occupied-general generalized Fock matrix elements
        tot_gen_fock[: self.full_nocc] = 2 * (inact_fock_pi + act_fock_pi).T

        # add occupied-active block of generalized Fock matrix
        tot_gen_fock[self.full_nocc :] = tot_targets.gen_fock

        return tot_targets.energy, tot_gen_fock

    def plot_results(self, *args: str) -> matplotlib.figure.Figure:
        """
        this function plots the generalized Fock matrix elements
        """
        raise NotImplementedError

    def free_ints(self) -> None:
        """
        this function deallocates integrals in shared memory after the calculation is
        done
        """
        super(GenFockExpCls, self).free_ints()

        # free additional integrals
        self.inact_fock_win.Free()
        self.eri_goaa_win.Free()
        self.eri_gaao_win.Free()
        self.eri_gaaa_win.Free()

        return

    def _int_wins(
        self,
        mpi: MPICls,
        hcore: Optional[np.ndarray],
        eri: Optional[np.ndarray],
        inact_fock: Optional[np.ndarray] = None,
        eri_goaa: Optional[np.ndarray] = None,
        eri_gaao: Optional[np.ndarray] = None,
        eri_gaaa: Optional[np.ndarray] = None,
        **kwargs: Optional[np.ndarray],
    ):
        """
        this function creates shared memory windows for integrals on every node
        """
        super()._int_wins(mpi, hcore, eri)

        # allocate additional integrals in shared mem
        self.inact_fock_win: MPI.Win = MPI.Win.Allocate_shared(
            (
                8 * self.full_norb * (self.full_nocc + self.norb)
                if mpi.local_master
                else 0
            ),
            8,
            comm=mpi.local_comm,
        )
        self.eri_goaa_win: MPI.Win = MPI.Win.Allocate_shared(
            (
                8 * self.full_norb * self.full_nocc * self.norb**2
                if mpi.local_master
                else 0
            ),
            8,
            comm=mpi.local_comm,
        )
        self.eri_gaao_win: MPI.Win = MPI.Win.Allocate_shared(
            (
                8 * self.full_norb * self.norb**2 * self.full_nocc
                if mpi.local_master
                else 0
            ),
            8,
            comm=mpi.local_comm,
        )
        self.eri_gaaa_win: MPI.Win = MPI.Win.Allocate_shared(
            8 * self.full_norb * self.norb**3 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,
        )

        # open additional integrals in shared memory
        self.inact_fock: np.ndarray = open_shared_win(
            self.inact_fock_win,
            np.float64,
            (self.full_norb, self.full_nocc + self.norb),
        )
        self.eri_goaa: np.ndarray = open_shared_win(
            self.eri_goaa_win,
            np.float64,
            (self.full_norb, self.full_nocc, self.norb, self.norb),
        )
        self.eri_gaao: np.ndarray = open_shared_win(
            self.eri_gaao_win,
            np.float64,
            (self.full_norb, self.norb, self.norb, self.full_nocc),
        )
        self.eri_gaaa: np.ndarray = open_shared_win(
            self.eri_gaaa_win,
            np.float64,
            (self.full_norb, self.norb, self.norb, self.norb),
        )

        # set additional integrals on global master
        if mpi.global_master:
            self.inact_fock[:] = cast(np.ndarray, inact_fock)
            self.eri_goaa[:] = cast(np.ndarray, eri_goaa)
            self.eri_gaao[:] = cast(np.ndarray, eri_gaao)
            self.eri_gaaa[:] = cast(np.ndarray, eri_gaaa)

        # mpi_bcast additional integrals
        if mpi.num_masters > 1 and mpi.local_master:
            self.inact_fock[:] = mpi_bcast(mpi.master_comm, self.inact_fock)
            self.eri_goaa[:] = mpi_bcast(mpi.master_comm, self.eri_goaa)
            self.eri_gaao[:] = mpi_bcast(mpi.master_comm, self.eri_gaao)
            self.eri_gaaa[:] = mpi_bcast(mpi.master_comm, self.eri_gaaa)

        # mpi barrier
        mpi.global_comm.Barrier()

        return

    @staticmethod
    def _convert_to_target(prop: Tuple[float, np.ndarray, np.ndarray]) -> GenFockCls:
        """
        this function converts the input target type into the used target type
        """
        return GenFockCls(*prop)

    def _calc_hf_prop(
        self, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
    ) -> GenFockCls:
        """
        this function calculates the hartree-fock generalized fock matrix
        """
        # add one-electron integrals
        hf_energy = np.sum(self.occup * np.diag(hcore))

        # set closed- and open-shell indices
        cs_idx = np.where(self.occup == 2)[0]
        os_idx = np.where(self.occup == 1)[0]

        # add closed-shell and coupling electron repulsion terms
        hf_energy += np.trace((np.sum(vhf, axis=0))[cs_idx.reshape(-1, 1), cs_idx])

        # check if system is open-shell
        if os_idx.size > 0:
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

        # initialize rdm1
        hf_rdm1 = np.zeros(2 * (self.norb,), dtype=np.float64)

        # set diagonal to occupation numbers
        np.einsum("ii->i", hf_rdm1)[...] += self.occup

        # get alpha and beta occupation vectors
        occup_a = self.occup.copy()
        occup_a[occup_a > 0.0] = 1.0
        occup_b = self.occup - occup_a

        # calculate occupied-active generalized Fock matrix elements
        hf_gen_fock = (
            np.einsum("u,pu->up", self.occup, self.inact_fock[:, self.full_nocc :])
            + np.einsum("u,v,puvv->up", self.occup, self.occup, self.eri_gaaa)
            - (
                np.einsum("u,v,pvvu->up", occup_a, occup_a, self.eri_gaaa)
                + np.einsum("u,v,pvvu->up", occup_b, occup_b, self.eri_gaaa)
            )
        )

        return GenFockCls(hf_energy, hf_rdm1, hf_gen_fock)

    def _ref_results(self, ref_prop: GenFockCls) -> str:
        """
        this function prints reference space results for a target calculation
        """
        energy = (
            f"reference space energy for root {self.fci_state_root} "
            + f"(total increment = {ref_prop.energy:.4e})"
        )
        header_rdm1 = f"reference space 1-particle RDM for root {self.fci_state_root}"
        rdm1 = f"(total increment norm = {np.linalg.norm(ref_prop.rdm1):.4e})"
        header_gen_fock = (
            f"reference space generalized Fock matrix for root {self.fci_state_root}"
        )
        gen_fock = f"(total increment norm = {np.linalg.norm(ref_prop.gen_fock):.4e})"

        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {energy:^80}\n"
        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {header_rdm1:^80}\n"
        string += f" RESULT: {rdm1:^80}\n"
        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {header_gen_fock:^80}\n"
        string += f" RESULT: {gen_fock:^80}\n"
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
    ) -> GenFockCls:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # perform main calc
        gen_fock = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            gen_fock -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        # get reference space indices in cas_idx
        idx = np.where(np.isin(cas_idx, self.ref_space))[0]
        gen_fock_idx = np.concatenate(
            (np.arange(self.nocc), self.ref_space[self.nocc <= self.ref_space])
        )

        # subtract reference space properties
        gen_fock[idx, gen_fock_idx] -= self.ref_prop

        return gen_fock

    def _sum(
        self,
        inc: List[List[packedGenFockCls]],
        hashes: List[List[np.ndarray]],
        tup: np.ndarray,
        tup_clusters: Optional[List[np.ndarray]],
    ) -> GenFockCls:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """
        # size of cas
        cas_size = self.ref_space.size + tup.size

        # number of occupied orbitals outside cas space
        ncore = self.nocc - self.ref_occ.size - (tup < self.nocc).sum()

        # init res
        res = GenFockCls(
            0.0,
            np.zeros((cas_size, cas_size), dtype=np.float64),
            np.zeros((ncore + cas_size, self.full_norb), dtype=np.float64),
        )

        # rank of reference space and occupied and virtual tuple orbitals
        rank = np.argsort(np.argsort(np.concatenate((self.ref_space, tup))))
        ref_idx = rank[: self.ref_space.size]
        ref_virt_idx = rank[self.ref_occ.size : self.ref_space.size]
        exp_idx = rank[self.ref_space.size :]

        # get cluster indices in tuple space
        exp_clusters_idx: Optional[List[np.ndarray]]
        if tup_clusters is not None:
            cluster_idx = 0
            exp_clusters_idx = []
            for cluster in tup_clusters:
                exp_clusters_idx.append(
                    ref_idx[cluster_idx : cluster_idx + cluster.size]
                )
                cluster_idx += cluster.size
        else:
            exp_clusters_idx = None

        # compute contributions from lower-order increments
        for k in range(self.order - 1, self.min_order - 1, -1):
            # rank of all orbitals in casci space
            idx_casci = np.empty(self.ref_space.size + k, dtype=np.int64)

            # loop over number of occupied orbitals
            for l in range(k + 1):
                # check if hashes are available
                if hashes[k - self.min_order][l].size > 0:
                    # indices of generalized Fock matrix subspace in full space
                    idx_gen_fock = np.empty(
                        self.nocc + self.ref_virt.size + k - l, dtype=np.int64
                    )

                    # add all occupied orbitals
                    idx_gen_fock[: self.nocc] = np.arange(self.nocc)

                    # loop over subtuples
                    for tup_sub, idx_sub, idx_sub_virt in tuples_idx_virt_idx_with_nocc(
                        tup,
                        tup_clusters,
                        exp_idx,
                        exp_clusters_idx,
                        self.nocc,
                        k,
                        l,
                        cached=True,
                    ):
                        # compute index
                        idx = hash_lookup(
                            hashes[k - self.min_order][l], hash_1d(tup_sub)
                        )

                        # sum up order increments
                        if idx is not None:
                            # add rank of reference space orbitals
                            idx_casci[: self.ref_space.size] = ref_idx

                            # add rank of subtuple orbitals
                            idx_casci[self.ref_space.size :] = idx_sub

                            # add rank of reference space virtual orbitals
                            idx_gen_fock[self.nocc : self.nocc + self.ref_virt.size] = (
                                ncore + ref_virt_idx
                            )

                            # add rank of subtuple virtual orbitals
                            idx_gen_fock[self.nocc + self.ref_virt.size :] = (
                                ncore + idx_sub_virt
                            )

                            # sort indices for faster assignment
                            idx_casci.sort()
                            idx_gen_fock[self.nocc :].sort()

                            # add subtuple rdms
                            res[idx_casci, idx_gen_fock] += inc[k - self.min_order][l][
                                idx
                            ]

                        else:
                            raise RuntimeError("Subtuple not found:", tup_sub)

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
    ) -> GenFockCls:
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
        e_cc, ccsd, eris = self._ccsd_driver_pyscf(
            h1e, h2e, core_idx, cas_idx, spin_cas, converge_amps=True
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

        # calculate generalized Fock matrix elements
        gen_fock = self._calc_gen_fock(core_idx, cas_idx, rdm1, rdm2)

        # get generalized Fock matrix indices
        gen_fock_idx = np.sort(np.concatenate((core_idx, cas_idx)))

        # get hartree-fock property
        hf_prop = self.hf_prop[cas_idx, gen_fock_idx]

        return GenFockCls(e_cc, rdm1, gen_fock) - hf_prop

    def _calc_gen_fock(
        self,
        occ_idx: np.ndarray,
        cas_idx: np.ndarray,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
    ):
        """
        this function calculates generalized Fock matrix elements from 1- and 2-particle
        reduced density matrices of a given CAS
        """
        # get indices in generalized Fock matrix subspace
        gen_fock_sort_idx = np.argsort(np.argsort(np.concatenate((occ_idx, cas_idx))))
        gen_fock_occ_idx = gen_fock_sort_idx[: occ_idx.size]
        gen_fock_cas_idx = gen_fock_sort_idx[occ_idx.size :]

        # initialize generalized Fock matrix
        gen_fock = np.empty(
            (occ_idx.size + cas_idx.size, self.full_norb), dtype=np.float64
        )

        # add inactive Fock matrix for orbitals inside CAS but outside iCAS
        inact_fock_pi = self.inact_fock[:, self.full_nocc + occ_idx]

        # add occupied orbitals inside CAS but outside iCAS to inactive Fock matrix
        # elements for occupied orbitals inside CAS but outside iCAS
        eri_pmn1 = self.eri_gaaa[:, occ_idx.reshape(-1, 1), occ_idx, occ_idx]
        eri_pmn2 = self.eri_gaaa[
            :, occ_idx.reshape(-1, 1), occ_idx.reshape(-1, 1), occ_idx
        ]
        inact_fock_pi += np.sum(2 * eri_pmn1 - eri_pmn2.transpose(0, 2, 1), axis=2)

        # calculate active Fock matrix elements
        eri_pmuv = self.eri_gaaa[
            :, occ_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), cas_idx
        ]
        eri_puvm = self.eri_gaaa[
            :, cas_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), occ_idx
        ]
        act_fock_pi = np.einsum(
            "uv,pmuv->pm", rdm1, eri_pmuv - 0.5 * eri_puvm.transpose(0, 3, 2, 1)
        )

        # calculate occupied-general generalized Fock matrix elements
        gen_fock[gen_fock_occ_idx] = 2 * (inact_fock_pi + act_fock_pi).T

        # initialize inactive Fock matrix for all active orbitals inside iCAS, these
        # now include all contributions from occupied orbitals outside CAS
        inact_fock_pu = self.inact_fock[:, self.full_nocc + cas_idx]

        # add occupied orbitals inside CAS but outside iCAS to inactive Fock matrix
        # elements for active orbitals inside iCAS
        eri_pum = self.eri_gaaa[:, cas_idx.reshape(-1, 1), occ_idx, occ_idx]
        eri_pmu = self.eri_gaaa[
            :, occ_idx.reshape(-1, 1), occ_idx.reshape(-1, 1), cas_idx
        ]
        inact_fock_pu += 2 * np.sum(eri_pum, axis=2) - np.sum(eri_pmu, axis=1)

        # calculate active-general generalized Fock matrix elements
        gen_fock[gen_fock_cas_idx] = np.einsum(
            "uv,pv->up", rdm1, inact_fock_pu
        ) + np.einsum(
            "uvxy,pvxy->up",
            rdm2,
            self.eri_gaaa[
                :, cas_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), cas_idx
            ],
        )

        return gen_fock

    @staticmethod
    def _write_target_file(prop: GenFockCls, string: str, order: int) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file_mult(
            {
                "energy": np.array(prop.energy),
                "rdm1": prop.rdm1,
                "gen_fock": prop.gen_fock,
            },
            string,
            order,
        )

    @staticmethod
    def _read_target_file(file: str) -> GenFockCls:
        """
        this function reads files of attributes with the target type
        """
        target_dict = np.load(os.path.join(RST, file))
        return GenFockCls(
            target_dict["energy"], target_dict["rdm1"], target_dict["gen_fock"]
        )

    def _init_target_inst(self, value: float, tup_norb: int, tup_nocc) -> GenFockCls:
        """
        this function initializes an instance of the target type
        """
        return GenFockCls(
            value,
            np.full(2 * (tup_norb,), value, dtype=np.float64),
            np.full(
                (self.nocc + tup_norb - tup_nocc, self.full_norb),
                value,
                dtype=np.float64,
            ),
        )

    @staticmethod
    def _mpi_reduce_target(
        comm: MPI.Comm, values: GenFockCls, op: MPI.Op
    ) -> GenFockCls:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return GenFockCls(
            mpi_reduce(
                comm, np.array(values.energy, dtype=np.float64), root=0, op=op
            ).item(),
            mpi_reduce(comm, values.rdm1, root=0, op=op),
            mpi_reduce(comm, values.gen_fock, root=0, op=op),
        )

    @staticmethod
    def _write_inc_file(inc: packedGenFockCls, order: int, nocc: int) -> None:
        """
        this function defines how to write increment restart files
        """
        np.savez(
            os.path.join(RST, f"mbe_inc_{order}_{nocc}"),
            energy=inc.energy,
            rdm1=inc.rdm1,
            gen_fock=inc.gen_fock,
        )

    @staticmethod
    def _read_inc_file(file: str) -> packedGenFockCls:
        """
        this function defines reads the increment restart files
        """
        target_dict = np.load(os.path.join(RST, file))

        return packedGenFockCls(
            target_dict["energy"], target_dict["rdm1"], target_dict["gen_fock"]
        )

    def _allocate_shared_inc(
        self,
        size: int,
        allocate: bool,
        comm: MPI.Intracomm,
        tup_norb: int,
        tup_nocc: int,
    ) -> Optional[Tuple[MPI.Win, MPI.Win, MPI.Win]]:
        """
        this function allocates a shared increment window
        """
        # generate packing and unpacking indices if these have not been generated yet
        if len(packedGenFockCls.rdm1_size) == tup_norb - 1:
            packedGenFockCls.get_pack_idx(self.ref_space.size + tup_norb)

        # get number of orbitals for generalized Fock matrix
        gen_fock_norb = (
            self.nocc + self.ref_space.size - self.ref_occ.size + tup_norb - tup_nocc
        )

        return (
            (
                MPI.Win.Allocate_shared(8 * size if allocate else 0, 8, comm=comm),
                MPI.Win.Allocate_shared(
                    (
                        8 * size * packedGenFockCls.rdm1_size[tup_norb - 1]
                        if allocate
                        else 0
                    ),
                    8,
                    comm=comm,
                ),
                MPI.Win.Allocate_shared(
                    8 * size * gen_fock_norb * self.full_norb if allocate else 0,
                    8,
                    comm=comm,
                ),
            )
            if size > 0
            else None
        )

    def _open_shared_inc(
        self,
        window: Optional[Tuple[MPI.Win, MPI.Win, MPI.Win]],
        n_incs: int,
        tup_norb: int,
        tup_nocc: int,
    ) -> packedGenFockCls:
        """
        this function opens a shared increment window
        """
        # get number of orbitals for generalized Fock matrix
        gen_fock_norb = (
            self.nocc + self.ref_space.size - self.ref_occ.size + tup_norb - tup_nocc
        )

        # open shared windows
        if window is not None:
            energy = open_shared_win(window[0], np.float64, (n_incs,))
            rdm1 = open_shared_win(
                window[1],
                np.float64,
                (n_incs, packedGenFockCls.rdm1_size[tup_norb - 1]),
            )
            gen_fock = open_shared_win(
                window[2],
                np.float64,
                (n_incs, gen_fock_norb, self.full_norb),
            )
        else:
            energy = np.empty(shape=n_incs, dtype=np.float64)
            rdm1 = np.empty(
                shape=(n_incs, packedGenFockCls.rdm1_size[tup_norb - 1]),
                dtype=np.float64,
            )
            gen_fock = np.empty(
                shape=(n_incs, gen_fock_norb, self.full_norb), dtype=np.float64
            )

        return packedGenFockCls(energy, rdm1, gen_fock, tup_norb - 1)

    def _init_inc_arr_from_lst(
        self, inc_lst: List[GenFockCls], tup_norb: int, tup_nocc: int
    ) -> packedGenFockCls:
        """
        this function creates an increment array from a list of increments
        """
        # get number of orbitals for generalized Fock matrix
        gen_fock_norb = (
            self.nocc + self.ref_space.size - self.ref_occ.size + tup_norb - tup_nocc
        )

        # initialize arrays
        energy = np.empty(len(inc_lst), dtype=np.float64)
        rdm1 = np.empty(
            (len(inc_lst), packedGenFockCls.rdm1_size[-1]), dtype=np.float64
        )
        gen_fock = np.empty(
            (len(inc_lst), gen_fock_norb, self.full_norb),
            dtype=np.float64,
        )

        # fill arrays
        for i, inc in enumerate(inc_lst):
            energy[i] = inc.energy
            rdm1[i] = inc.rdm1[packedGenFockCls.pack_rdm1[-1]]
            gen_fock[i] = inc.gen_fock

        return packedGenFockCls(energy, rdm1, gen_fock)

    @staticmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: packedGenFockCls) -> packedGenFockCls:
        """
        this function bcasts the increments
        """
        return packedGenFockCls(
            mpi_bcast(comm, inc.energy),
            mpi_bcast(comm, inc.rdm1),
            mpi_bcast(comm, inc.gen_fock),
        )

    @staticmethod
    def _mpi_reduce_inc(
        comm: MPI.Comm, inc: packedGenFockCls, op: MPI.Op
    ) -> packedGenFockCls:
        """
        this function performs a MPI reduce operation on the increments
        """
        return packedGenFockCls(
            mpi_reduce(comm, inc.energy, root=0, op=op),
            mpi_reduce(comm, inc.rdm1, root=0, op=op),
            mpi_reduce(comm, inc.gen_fock, root=0, op=op),
        )

    @staticmethod
    def _mpi_allreduce_inc(
        comm: MPI.Comm, inc: packedGenFockCls, op: MPI.Op
    ) -> packedGenFockCls:
        """
        this function performs a MPI allreduce operation on the increments
        """
        return packedGenFockCls(
            mpi_allreduce(comm, inc.energy, op=op),
            mpi_allreduce(comm, inc.rdm1, op=op),
            mpi_allreduce(comm, inc.gen_fock, op=op),
        )

    @staticmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: packedGenFockCls, recv_inc: Optional[packedGenFockCls]
    ) -> None:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # size of arrays on every rank
        counts_dict = {
            "energy": comm.gather(send_inc.energy.size),
            "rdm1": comm.allgather(send_inc.rdm1.size),
            "gen_fock": comm.gather(send_inc.gen_fock.size),
        }

        # receiving arrays
        recv_inc_dict: Dict[str, Optional[Tuple[np.ndarray, List[int]]]] = {}
        if recv_inc is not None:
            if (
                counts_dict["energy"] is not None
                and counts_dict["rdm1"] is not None
                and counts_dict["gen_fock"] is not None
            ):
                recv_inc_dict["energy"] = (recv_inc.energy, counts_dict["energy"])
                recv_inc_dict["rdm1"] = (recv_inc.rdm1, counts_dict["rdm1"])
                recv_inc_dict["gen_fock"] = (recv_inc.gen_fock, counts_dict["gen_fock"])
            else:
                recv_inc_dict["energy"] = recv_inc_dict["rdm1"] = recv_inc_dict[
                    "gen_fock"
                ] = None
        else:
            recv_inc_dict["energy"] = recv_inc_dict["rdm1"] = recv_inc_dict[
                "gen_fock"
            ] = None

        comm.Gatherv(send_inc.energy, recv_inc_dict["energy"], root=0)
        comm.Gatherv(send_inc.rdm1.ravel(), recv_inc_dict["rdm1"], root=0)
        comm.Gatherv(send_inc.gen_fock.ravel(), recv_inc_dict["gen_fock"], root=0)

    @staticmethod
    def _free_inc(inc_win: Tuple[MPI.Win, MPI.Win, MPI.Win]) -> None:
        """
        this function frees the supplied increment windows
        """
        inc_win[0].Free()
        inc_win[1].Free()
        inc_win[2].Free()

    def _add_screen(
        self,
        inc_tup: Union[GenFockCls, packedGenFockCls],
        order: int,
        tup: np.ndarray,
        *args: Optional[List[np.ndarray]],
    ) -> None:
        """
        this function modifies the screening array
        """
        self.screen[order - 1]["max"][tup] = np.maximum(
            self.screen[order - 1]["max"][tup], np.max(np.abs(inc_tup.rdm1))
        )
        self.screen[order - 1]["sum_abs"][tup] += np.sum(np.abs(inc_tup.gen_fock))

    def _update_inc_stats(
        self,
        inc_tup: GenFockCls,
        min_inc: GenFockCls,
        mean_inc: GenFockCls,
        max_inc: GenFockCls,
        cas_idx: np.ndarray,
        n_eqv_tups: int,
    ) -> Tuple[GenFockCls, GenFockCls, GenFockCls]:
        """
        this function updates the increment statistics
        """
        # get indices for generalized Fock matrix
        gen_fock_idx = np.concatenate(
            (np.arange(self.nocc), cas_idx[self.nocc <= cas_idx])
        )

        # add to total rdm
        mean_inc[cas_idx, gen_fock_idx] += inc_tup

        return min_inc, mean_inc, max_inc

    def _add_prop(
        self, prop_tup: GenFockCls, tot_prop: GenFockCls, cas_idx: np.ndarray
    ) -> GenFockCls:
        """
        this function adds a tuple property to the property of the full space
        """
        # get indices for generalized Fock matrix
        gen_fock_idx = np.concatenate(
            (np.arange(self.nocc), cas_idx[self.nocc <= cas_idx])
        )

        tot_prop[cas_idx, gen_fock_idx] += prop_tup
        return tot_prop

    @staticmethod
    def _total_inc(inc: List[packedGenFockCls], mean_inc: GenFockCls) -> GenFockCls:
        """
        this function calculates the total increment at a certain order
        """
        return GenFockCls(
            np.sum(np.concatenate([item.energy for item in inc])),
            mean_inc.rdm1.copy(),
            mean_inc.gen_fock.copy(),
        )

    def _adaptive_screen(self, inc: List[List[packedGenFockCls]]):
        """
        this function wraps the adaptive screening function
        """
        raise NotImplementedError

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for a generalized Fock matrix
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

        # set headers
        energy = (
            f"energy for root {self.fci_state_root} "
            + f"(total increment = {tot_inc.energy:.4e})"
        )
        rdm1 = (
            f"1-rdm for root {self.fci_state_root} "
            + f"(total increment norm = {np.linalg.norm(tot_inc.rdm1):.4e})"
        )
        gen_fock = (
            f"generalized Fock matrix for root {self.fci_state_root} "
            + f"(total increment norm = {np.linalg.norm(tot_inc.gen_fock):.4e})"
        )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{energy:^81}\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{rdm1:^81}\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{gen_fock:^81}\n"

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


class ssGenFockExpCls(GenFockExpCls[int, np.ndarray]):
    """
    this class contains the pymbe expansion attributes for the generalized Fock matrix
    elements
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
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool = True,
    ) -> Tuple[GenFockCls, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # run fci calculation
        energy, civec, solver = self._fci_driver(
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

        # calculate 1- and 2-RDMs
        rdm1, rdm2 = solver.make_rdm12(civec[0], cas_idx.size, nelec)

        # calculate generalized Fock matrix elements
        gen_fock = self._calc_gen_fock(core_idx, cas_idx, rdm1, rdm2)

        # get indices for generalized Fock matrix
        gen_fock_idx = np.sort(np.concatenate((core_idx, cas_idx)))

        # get hartree-fock property
        hf_prop = self.hf_prop[cas_idx, gen_fock_idx]

        return GenFockCls(energy[-1], rdm1, gen_fock) - hf_prop, civec

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: GenFockCls,
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
            f"      energy increment for root {self.fci_state_root:d} "
            + f"= {inc_tup.energy:.4e}\n"
        )
        if logger.getEffectiveLevel() == 10:
            string += (
                f"      rdm1 increment for root {self.fci_state_root:d} = "
                + np.array2string(inc_tup.rdm1, max_line_width=59, precision=4)
                + "\n"
            )
            string += (
                f"      generalized Fock matrix increment for root "
                + "{self.fci_state_root:d} = "
                + np.array2string(inc_tup.gen_fock, max_line_width=59, precision=4)
                + "\n"
            )

        return string


class saGenFockExpCls(GenFockExpCls[List[int], List[np.ndarray]]):
    """
    this class contains the pymbe expansion attributes for the generalized Fock matrix
    elements
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
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        _nelec: np.ndarray,
        ref_guess: bool = True,
    ) -> Tuple[GenFockCls, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # initialize state-averaged energy
        sa_energy = 0.0

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
            energy, civec, solver = self._fci_driver(
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

            # calculate state-averaged energy and 1- and 2-RDMs
            for root, state_idx in zip(roots, solver_info["states"]):
                sa_energy += self.fci_state_weights[state_idx] * energy[root]
                sa_rdm12 += self.fci_state_weights[state_idx] * RDMCls(
                    *solver.make_rdm12(civec[root], cas_idx.size, solver_info["nelec"])
                )

        # calculate generalized Fock matrix elements
        sa_gen_fock = self._calc_gen_fock(
            core_idx, cas_idx, sa_rdm12.rdm1, sa_rdm12.rdm2
        )

        # get indices for generalized Fock matrix
        gen_fock_idx = np.sort(np.concatenate((core_idx, cas_idx)))

        # get hartree-fock property
        hf_prop = self.hf_prop[cas_idx, gen_fock_idx]

        return GenFockCls(sa_energy, sa_rdm12.rdm1, sa_gen_fock) - hf_prop, civec

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: GenFockCls,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        if logger.getEffectiveLevel() == 10:
            string += (
                f"      energy increment for averaged states = {inc_tup.energy:.4e}\n"
            )
            string += (
                f"      rdm1 increment for averaged states = "
                + np.array2string(inc_tup.rdm1, max_line_width=59, precision=4)
                + "\n"
            )
            string += (
                f"      generalized Fock matrix increment for averaged states = "
                + np.array2string(inc_tup.gen_fock, max_line_width=59, precision=4)
                + "\n"
            )

        return string
