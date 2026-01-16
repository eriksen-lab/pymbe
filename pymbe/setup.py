#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import sys
import os
import numpy as np
from json import load, dump
from pyscf import symm
from typing import TYPE_CHECKING

from pymbe.logger import logger
from pymbe.parallel import kw_dist, system_dist
from pymbe.tools import RST, TupSqOverlapType, logger_config, ground_state_sym
from pymbe.output import main_header, ref_space_results


if TYPE_CHECKING:
    from pymbe.pymbe import MBE, Tuple, List, Optional


def general_setup(mbe: MBE):
    """
    this function performs the general setup at the start of every calculation
    """
    # configure logging
    logger_config(mbe.verbose)

    # sanity check
    sanity_check(mbe)

    # print expansion headers
    logger.info(main_header(mbe.mpi))

    # dump flags
    for key, value in vars(mbe).items():
        if key in [
            "mol",
            "hcore",
            "eri",
            "mpi",
            "exp",
            "dipole_ints",
            "inact_fock",
            "eri_goaa",
            "eri_gaao",
            "eri_gaaa",
        ]:
            logger.debug(f"   -- {key:<15}: {str(value)}")
        else:
            logger.info(f"   -- {key:<15}: {str(value)}")
    logger.debug("")
    for key, value in vars(mbe.mpi).items():
        logger.debug(" " + key + " = " + str(value))
    logger.info("")
    logger.info("")

    # method
    logger.info(f"{('-' * 45):^87}")
    logger.info(f"{mbe.method.upper() + ' expansion':^87s}")
    logger.info(f"{('-' * 45):^87}")

    # write restart files
    if mbe.rst and not mbe.restarted:
        # create restart folder
        if not os.path.isdir(RST):
            os.mkdir(RST)

        # write keywords
        restart_write_kw(mbe)

        # write system quantities
        restart_write_system(mbe)


def clustering_setup(mbe: MBE, max_cluster_size: int):
    """
    this function performs the setup for clustering
    """
    # header
    logger.info(" " + 92 * "-" + "\n " + 92 * "|" + "\n " + 92 * "-")
    logger.info(
        f" Determining orbital clusters of maximum size {max_cluster_size} by "
        f"considering early-order MBE information"
    )
    logger.info(" " + 92 * "-")

    # configure logging
    logger_config(mbe.verbose)

    # sanity check
    sanity_check(mbe)
    if (
        isinstance(mbe.nelec, list)
        and isinstance(mbe.fci_state_sym, list)
        and isinstance(mbe.fci_state_root, list)
        and isinstance(mbe.fci_state_weights, list)
    ):
        raise ValueError(
            "orbital clustering not possible for state-averaged properties, "
            "calculation must only have keywords describing a single state and can "
            "therefore not be lists (nelec, fci_state_sym, fci_state_root, "
            "fci_state_weights keyword arguments)"
        )

    # write restart files
    if mbe.rst and not mbe.restarted:
        # create restart folder
        os.mkdir(RST)

        # write keywords
        restart_write_kw(mbe)

        # write system quantities
        restart_write_system(mbe)


def sanity_check(mbe: MBE) -> None:
    """
    this function performs sanity checks of all mbe attributes
    """
    # general settings
    if not 3 <= sys.version_info[0]:
        raise RuntimeError("PyMBE only runs under python3+")
    if not int(os.environ.get("PYTHONHASHSEED", -1)) == 0:
        raise ValueError("environment variable PYTHONHASHSEED must be set to zero")

    # expansion model
    if not isinstance(mbe.method, str):
        raise TypeError(
            "electronic structure method (method keyword argument) must be a string"
        )
    if mbe.method not in ["ccsd", "ccsd(t)", "ccsdt", "ccsdt(q)", "ccsdtq", "fci"]:
        raise ValueError(
            "valid electronic structure methods (method keyword argument) are: ccsd, "
            "ccsd(t), ccsdt, ccsdt(q), ccsdtq and fci"
        )

    # targets
    if not isinstance(mbe.target, str):
        raise TypeError(
            "expansion target property (target keyword argument) must be a string"
        )
    if mbe.target not in [
        "energy",
        "excitation",
        "dipole",
        "trans",
        "rdm12",
        "genfock",
    ]:
        raise ValueError(
            "invalid choice for target property (target keyword argument). valid "
            "choices are: energy, excitation energy (excitation), dipole, transition "
            "dipole (trans), 1- and 2-particle reduced density matrices (rdm12) and "
            "generalized Fock matrix (genfock)"
        )
    if mbe.method != "fci":
        if mbe.target not in ["energy", "dipole", "rdm12", "genfock"]:
            raise ValueError(
                "excited target states (target keyword argument) not implemented for "
                "chosen expansion model (method keyword argument)"
            )

    # system
    if not (hasattr(mbe, "norb") and isinstance(mbe.norb, int)):
        raise TypeError("number of orbitals (norb keyword argument) must be an int")
    if mbe.norb <= 0:
        raise ValueError("number of orbitals (norb keyword argument) must be > 0")
    if not (
        (
            hasattr(mbe, "nelec")
            and (isinstance(mbe.nelec, np.ndarray) and mbe.nelec.dtype == np.int64)
            or (
                isinstance(mbe.nelec, list)
                and all(
                    [
                        isinstance(state, np.ndarray) and state.dtype == np.int64
                        for state in mbe.nelec
                    ]
                )
            )
        )
    ):
        raise TypeError(
            "number of electrons (nelec keyword argument) must be an int, a tuple of "
            "ints or a np.ndarray of ints (state-specific) or a list of tuples of ints "
            "or a list of np.ndarray of ints (state-averaged)"
        )
    if isinstance(mbe.nelec, np.ndarray) and (
        mbe.nelec.size != 2 or (mbe.nelec[0] <= 0 and mbe.nelec[1] <= 0)
    ):
        raise ValueError(
            "number of electrons (nelec keyword argument) for state-specific "
            "calculations must be an int > 0 or a tuple or np.ndarray of ints with "
            "dimension 2 and at least one element > 0"
        )
    if isinstance(mbe.nelec, list) and any(
        [state.size != 2 or (state[0] <= 0 and state[1] <= 0) for state in mbe.nelec]
    ):
        raise ValueError(
            "number of electrons (nelec keyword argument) for state-averaged "
            "calculations must be a list of tuples or np.ndarrays of ints with "
            "dimension 2 and at least one element > 0"
        )
    if not isinstance(mbe.point_group, str):
        raise TypeError("symmetry (point_group keyword argument) must be a str")
    if not (
        hasattr(mbe, "orbsym")
        and (isinstance(mbe.orbsym, np.ndarray) and mbe.orbsym.dtype == np.int64)
        or (
            isinstance(mbe.orbsym, list)
            and all([isinstance(symm_op, list) for symm_op in mbe.orbsym])
            and all(
                [
                    [
                        isinstance(tup, tuple)
                        and len(tup) == 2
                        and isinstance(tup[0], tuple)
                        and isinstance(tup[1], tuple)
                        for tup in symm_op
                    ]
                    for symm_op in mbe.orbsym
                ]
            )
            and all(
                [
                    [
                        [isinstance(orb, int) for orb in tup[0]]
                        + [isinstance(orb, int) for orb in tup[1]]
                        for tup in symm_op
                    ]
                    for symm_op in mbe.orbsym
                ]
            )
        )
    ):
        raise TypeError(
            "orbital symmetry (orbsym keyword argument) must be a np.ndarray of ints "
            "or a list of symmetry operation lists of permutation tuples of two tuples "
            "with orbital index ints"
        )
    if isinstance(mbe.orbsym, np.ndarray) and mbe.orbsym.shape != (mbe.norb,):
        raise ValueError(
            "orbital symmetry (orbsym keyword argument) must have shape (norb,)"
        )
    if not (
        isinstance(mbe.fci_state_sym, (str, int))
        or (
            isinstance(mbe.fci_state_sym, list)
            and all([isinstance(state, (str, int)) for state in mbe.fci_state_sym])
        )
    ):
        raise TypeError(
            "state wavefunction symmetry (fci_state_sym keyword argument) must be a "
            "str or int or a list of str or int"
        )
    try:
        if isinstance(mbe.fci_state_sym, str):
            mbe.fci_state_sym = symm.addons.irrep_name2id(
                mbe.point_group, mbe.fci_state_sym
            )
        elif isinstance(mbe.fci_state_sym, list) and all(
            [isinstance(state, str) for state in mbe.fci_state_sym]
        ):
            mbe.fci_state_sym = [
                symm.addons.irrep_name2id(mbe.point_group, state)
                for state in mbe.fci_state_sym
            ]
    except Exception as err:
        raise ValueError(
            "illegal choice of state wavefunction symmetry (fci_state_sym keyword "
            f"argument) -- PySCF error: {err}"
        )
    if not (
        (isinstance(mbe.fci_state_root, int))
        or (
            isinstance(mbe.fci_state_root, list)
            and all([isinstance(state, int) for state in mbe.fci_state_root])
        )
    ):
        raise TypeError(
            "target state (root keyword argument) must be an int (state-specific) or a "
            "list of ints (state-averaged)"
        )
    if isinstance(mbe.fci_state_root, int) and mbe.fci_state_root < 0:
        raise ValueError(
            "target state (root keyword argument) for state-specific calculations must "
            "be an int >= 0"
        )
    if isinstance(mbe.fci_state_root, list) and any(
        [state < 0 for state in mbe.fci_state_root]
    ):
        raise ValueError(
            "target state (root keyword argument) for state-averaged calculations must "
            "be a list of ints with all elements >= 0"
        )
    if (
        isinstance(mbe.nelec, np.ndarray)
        and isinstance(mbe.fci_state_sym, (str, int))
        and isinstance(mbe.fci_state_root, int)
        and not hasattr(mbe, "fci_state_weights")
    ):
        if isinstance(mbe.orbsym, np.ndarray):
            hf_wfnsym = ground_state_sym(mbe.orbsym, mbe.nelec, mbe.point_group)
            if (
                mbe.method != "fci" or mbe.base_method is not None
            ) and mbe.fci_state_sym != hf_wfnsym:
                raise ValueError(
                    "illegal choice of fci state wavefunction symmetry (fci_state_sym "
                    "keyword argument) for chosen expansion model (method or "
                    "base_method keyword argument) because fci_state_sym does not "
                    "equal hf state symmetry"
                )

        if (
            mbe.method != "fci" or mbe.base_method is not None
        ) and mbe.fci_state_root != 0:
            raise ValueError(
                "excited target states (root keyword argument) not implemented for "
                "chosen expansion model (method or base_method keyword argument)"
            )
        if mbe.target in ["excitation", "trans"] and not mbe.fci_state_root > 0:
            raise ValueError(
                "calculation of excitation energies or transition dipole moments "
                "(target keyword argument) requires target state root (state_root "
                "keyword argument) >= 1"
            )
    elif (
        isinstance(mbe.nelec, list)
        and isinstance(mbe.fci_state_sym, list)
        and isinstance(mbe.fci_state_root, list)
        and isinstance(mbe.fci_state_weights, list)
    ):
        if mbe.target not in ["rdm12", "genfock"]:
            raise ValueError(
                "only 1- and 2-particle reduced density matrices and generalized Fock "
                "matrices can be determined as state-averaged properties, all other "
                "mbe targets must only have keywords describing a single state and can "
                "therefore not be lists (nelec, fci_state_sym, fci_state_root, "
                "fci_state_weights keyword arguments)"
            )
        if mbe.method != "fci":
            raise ValueError(
                "only the fci method (method keyword argument) can be used to "
                "calculate state-averaged 1- and 2-particle reduced density matrices, "
                "cc methods must only have keywords describing a single state and can "
                "therefore not be lists (nelec, fci_state_sym, fci_state_root, "
                "fci_state_weights keyword arguments)"
            )
        if not (
            len(mbe.nelec)
            == len(mbe.fci_state_sym)
            == len(mbe.fci_state_root)
            == len(mbe.fci_state_weights)
        ):
            raise ValueError(
                "keywords describing different states for the calculation of "
                "state-averaged 1- and 2-particle reduced density matrices (nelec, "
                "fci_state_sym, fci_state_root, fci_state_weights keyword arguments) "
                "must all have the same length"
            )
        states = [
            state
            for state in zip(
                *[
                    [tuple(nelec_state) for nelec_state in mbe.nelec],
                    mbe.fci_state_sym,
                    mbe.fci_state_root,
                ]
            )
        ]
        if len(set(states)) != len(states):
            raise ValueError(
                "keywords describing multiple states for the calculation of "
                "state-averaged 1- and 2-particle reduced density matrices (nelec, "
                "fci_state_sym, fci_state_root keyword arguments) must describe "
                "different states, every state must therefore differ from every other "
                "state in either multiplicity, symmetry or root"
            )

    # orbital representation
    if not isinstance(mbe.orb_type, str):
        raise TypeError(
            "orbital representation (orb_type keyword argument) must be a string"
        )
    if mbe.orb_type not in ["can", "local", "ccsd", "ccsd(t)", "casscf"]:
        raise ValueError(
            "valid orbital representations (orb_type keyword argument) are currently: "
            "canonical (can), pipek-mezey (local), natural (ccsd or ccsd(t) or casscf "
            "orbs (casscf))"
        )
    if (
        mbe.orb_type == "local"
        and mbe.target in ["rdm12", "genfock"]
        and mbe.point_group != "C1"
    ):
        logger.warning(
            "Warning: 1- and 2-particle reduced density matrix and generalized Fock "
            "matrix calculations while exploiting local orbital symmetry are currently "
            "not possible. The symmetry of the local orbitals is not utilized in the "
            "current calculation."
        )

    # integrals
    if not (hasattr(mbe, "hcore") and isinstance(mbe.hcore, np.ndarray)):
        raise TypeError(
            "core hamiltonian integral (hcore keyword argument) must be a np.ndarray"
        )
    if mbe.hcore.shape != 2 * (mbe.norb,):
        raise ValueError(
            "core hamiltonian integral (hcore keyword argument) must have shape "
            "(norb, norb)"
        )
    if not (hasattr(mbe, "eri") and isinstance(mbe.eri, np.ndarray)):
        raise TypeError(
            "electron repulsion integral (eri keyword argument) must be a np.ndarray "
        )
    if mbe.eri.shape != 2 * (mbe.norb * (mbe.norb + 1) / 2,) and mbe.eri.shape != 4 * (
        mbe.norb,
    ):
        raise ValueError(
            "electron repulsion integral (eri keyword argument) must have shape "
            "(mbe.norb * (mbe.norb + 1) / 2, (mbe.norb * (mbe.norb + 1) / 2)) or "
            "(norb, norb, norb, norb)"
        )

    # reference and expansion spaces
    if not isinstance(mbe.ref_space, np.ndarray):
        raise TypeError(
            "reference space (ref_space keyword argument) must be a np.ndarray of "
            "orbital indices"
        )
    if (isinstance(mbe.nelec, np.ndarray) and mbe.nelec[0] != mbe.nelec[1]) or (
        isinstance(mbe.nelec, list)
        and any([state[0] != state[1] for state in mbe.nelec])
    ):
        if not np.all(
            np.isin(
                np.arange(np.amin(mbe.nelec), np.amax(mbe.nelec)),
                mbe.ref_space,
            )
        ):
            raise ValueError(
                "all partially occupied orbitals have to be included in the reference "
                "space (ref_space keyword argument)"
            )
    if not isinstance(mbe.exp_space, list) or not all(
        isinstance(cluster, np.ndarray) for cluster in mbe.exp_space
    ):
        raise TypeError(
            "expansion space (exp_space keyword argument) must be a np.ndarray of "
            "orbital indices or a list of np.ndarrays of orbital cluster indices"
        )
    exp_space = np.hstack(mbe.exp_space)
    if np.intersect1d(mbe.ref_space, exp_space).size != 0:
        raise ValueError(
            "reference space (ref_space keyword argument) and expansion space "
            "(exp_space keyword argument) must be mutually exclusive"
        )
    if np.unique(exp_space).size < exp_space.size:
        raise ValueError(
            "expansion space clusters (exp_space keyword argument) must be mutually "
            "exclusive"
        )
    if (
        not isinstance(mbe.ref_thres, float)
        or mbe.ref_thres < 0.0
        or mbe.ref_thres >= 1.0
    ):
        raise TypeError(
            "reference space squared overlap threshold (ref_thres keyword argument) "
            "must be a float >= 0.0 and < 1.0"
        )
    if mbe.ref_thres > 0.0 and mbe.method != "fci":
        raise ValueError(
            "automatic reference space generation (ref_thres keyword argument) is only "
            "possible with fci (method keyword argument)"
        )

    # base model
    if not isinstance(mbe.base_method, (str, type(None))):
        raise TypeError(
            "base model electronic structure method (base_method keyword argument) "
            "must be a str or None"
        )
    if mbe.base_method is not None:
        if mbe.base_method not in ["ccsd", "ccsd(t)", "ccsdt", "ccsdt(q)", "ccsdtq"]:
            raise ValueError(
                "valid base model electronic structure methods (base_method keyword "
                "argument) are currently: ccsd, ccsd(t), ccsdt, ccsdt(q) and ccsdtq"
            )
        if mbe.target not in ["energy", "dipole", "rdm12", "genfock"]:
            raise ValueError(
                "excited target states (target keyword argument) not implemented for "
                "base model calculations (base_method keyword argument)"
            )
        if mbe.fci_state_root != 0:
            raise ValueError(
                "excited target states (root keyword argument) not implemented for "
                "base model (base_method keyword argument)"
            )
        if mbe.target == "energy":
            if not (hasattr(mbe, "base_prop") and isinstance(mbe.base_prop, float)):
                raise TypeError(
                    "base model energy (base_prop keyword argument) must be a float"
                )
        elif mbe.target == "dipole":
            if not (
                hasattr(mbe, "base_prop") and isinstance(mbe.base_prop, np.ndarray)
            ):
                raise TypeError(
                    "base model dipole moment (base_prop keyword argument) must be a "
                    "np.ndarray"
                )
            if mbe.base_prop.shape != (3,):
                raise ValueError(
                    "base model dipole moment (base_prop keyword argument) must have "
                    "shape (3,)"
                )
        elif mbe.target == "rdm12":
            if not (hasattr(mbe, "base_prop") and isinstance(mbe.base_prop, tuple)):
                raise TypeError(
                    "base model 1- and 2-particle density matrices (base_prop keyword "
                    "argument) must be a tuple"
                )
            if len(mbe.base_prop) != 2:
                raise ValueError(
                    "base model 1- and 2-particle density matrices (base_prop keyword "
                    "argument) must have dimension 2"
                )
            if not (
                isinstance(mbe.base_prop[0], np.ndarray)
                and isinstance(mbe.base_prop[1], np.ndarray)
            ):
                raise TypeError(
                    "base model 1- and 2-particle density matrices (elements of "
                    "base_prop keyword argument) must be np.ndarrays"
                )
            if mbe.base_prop[0].shape != 2 * (mbe.norb,) or mbe.base_prop[
                1
            ].shape != 4 * (mbe.norb,):
                raise ValueError(
                    "base model 1- and 2-particle density matrices (elements of "
                    "base_prop keyword argument) must have shape (norb, norb) for the "
                    "rdm1 and shape (norb, norb, norb, norb) for the rdm2"
                )
        elif mbe.target == "genfock":
            if not (hasattr(mbe, "base_prop") and isinstance(mbe.base_prop, tuple)):
                raise TypeError(
                    "base model property (base_prop keyword argument) must be a tuple"
                )
            if len(mbe.base_prop) != 3:
                raise ValueError(
                    "base model property (base_prop keyword argument) must have "
                    "dimension 3"
                )
            if not (
                isinstance(mbe.base_prop[0], float)
                and isinstance(mbe.base_prop[1], np.ndarray)
                and isinstance(mbe.base_prop[2], np.ndarray)
            ):
                raise TypeError(
                    "first element of base model property (base_prop keyword argument) "
                    "describes the energy and must be a float, second element "
                    "describes the 1-particle RDM and must be a np.ndarray with shape "
                    "(norb, norb), third element describes the generalized fock matrix "
                    "and must be a np.ndarray "
                )
            if mbe.base_prop[1].shape != 2 * (mbe.norb,):
                raise ValueError(
                    "base model 1-particle RDM (second element of base_prop keyword "
                    "argument) must have shape (norb, norb)"
                )
            if mbe.base_prop[2].shape != (mbe.full_nocc + mbe.norb, mbe.full_norb):
                raise ValueError(
                    "base model generalized fock matrix (third element of base_prop "
                    "keyword argument) must have shape (full_nocc + norb, full_norb)"
                )

    # screening
    if not isinstance(mbe.screen_type, str):
        raise TypeError(
            "screening type (screen_type keyword argument) must be a string"
        )
    if mbe.screen_type not in ["fixed", "adaptive"]:
        raise ValueError(
            "valid screening types (screen_type keyword argument) are: fixed and "
            "adaptive"
        )
    if mbe.screen_type == "adaptive" and mbe.target != "energy":
        raise ValueError(
            "adaptive screening (screen_type keyword argument) is currently only "
            "implemented for energy expansions (target keyword argument)"
        )
    if not isinstance(mbe.screen_start, int):
        raise TypeError(
            "screening start order (screen_start keyword argument) must be an int"
        )
    if mbe.screen_start < 2:
        raise ValueError(
            "screening start order (screen_start keyword argument) must be >= 2"
        )
    if not isinstance(mbe.screen_perc, float):
        raise TypeError(
            "screening percentage (screen_perc keyword argument) must be a float"
        )
    if mbe.screen_perc > 1.0:
        raise ValueError(
            "screening percentage (screen_perc keyword argument) must be <= 1."
        )
    if not isinstance(mbe.screen_thres, float):
        raise TypeError(
            "screening threshold (screen_thres keyword argument) must be a float"
        )
    if mbe.screen_thres <= 0.0:
        raise ValueError(
            "screening threshold (screen_thres keyword argument) must be > 0."
        )
    if not isinstance(mbe.screen_func, str):
        raise TypeError(
            "screening function (screen_func keyword argument) must be an str"
        )
    if mbe.screen_func not in ["max", "sum_abs", "sum", "rnd"]:
        raise ValueError(
            "valid screening functions (screen_func keyword argument) are: max, "
            "sum_abs, sum and rnd"
        )
    if mbe.screen_func == "sum" and mbe.target not in ["energy", "excitation"]:
        raise ValueError(
            "screening with the sum screening function (screen_func keyword argument) "
            "only works for scalar targets such as energy or excitation (target "
            "keyword argument)"
        )
    if not isinstance(mbe.max_order, (int, type(None))):
        raise TypeError(
            "maximum expansion order (max_order keyword argument) must be an int or "
            "None"
        )
    if isinstance(mbe.max_order, int) and mbe.max_order < 1:
        raise ValueError(
            "maximum expansion order (max_order keyword argument) must be >= 1"
        )

    # restart
    if not isinstance(mbe.rst, bool):
        raise TypeError("restart logical (rst keyword argument) must be a bool")
    if not isinstance(mbe.rst_freq, int):
        raise TypeError("restart frequency (rst_freq keyword argument) must be an int")
    if mbe.rst_freq < 1:
        raise ValueError("restart frequency (rst_freq keyword argument) must be >= 1")

    # verbose
    if not isinstance(mbe.verbose, int):
        raise TypeError("verbose option (verbose keyword argument) must be an int")
    if mbe.verbose < 0:
        raise ValueError("verbose option (verbose keyword argument) must be  >= 0")

    # backends
    if mbe.method == "fci":
        if not isinstance(mbe.fci_backend, str):
            raise TypeError(
                "fci backend (fci_backend keyword argument) must be a string"
            )
        if mbe.fci_backend not in [
            "direct_spin0",
            "direct_spin1",
            "direct_spin0_symm",
            "direct_spin1_symm",
        ]:
            raise ValueError(
                "valid fci backends (fci_backend keyword argument) are: direct_spin0, "
                "direct_spin1, direct_spin0_symm, direct_spin1_symm"
            )
    if (
        mbe.method in ["ccsd", "ccsd(t)" "ccsdt", "ccsdt(q)", "ccsdtq"]
        or mbe.base_method is not None
    ):
        if not isinstance(mbe.cc_backend, str):
            raise TypeError(
                "coupled-cluster backend (cc_backend keyword argument) must be a "
                "string"
            )
        if mbe.cc_backend not in ["pyscf", "ecc", "ncc"]:
            raise ValueError(
                "valid coupled-cluster backends (cc_backend keyword argument) are: "
                "pyscf, ecc and ncc"
            )
        if mbe.method == "ccsdt" and mbe.cc_backend == "pyscf":
            raise ValueError(
                "ccsdt (method keyword argument) is not available with the pyscf "
                "coupled-cluster backend (cc_backend keyword argument)"
            )
        if mbe.method in ["ccsdt(q)", "ccsdtq"] and mbe.cc_backend != "ncc":
            raise ValueError(
                "ccsdt(q) and ccsdtq (method keyword argument) are not available with "
                "the pyscf and ecc coupled-cluster backends (cc_backend keyword "
                "argument)"
            )
        if mbe.cc_backend in ["ecc", "ncc"] and mbe.target != "energy":
            raise ValueError(
                "calculation of targets (target keyword argument) other than energy "
                "are not possible using the ecc and ncc backends (cc_backend keyword "
                "argument)"
            )
        if (isinstance(mbe.nelec, np.ndarray) and mbe.nelec[0] != mbe.nelec[1]) or (
            isinstance(mbe.nelec, list)
            and any([state[0] != state[1] for state in mbe.nelec])
        ):
            if mbe.cc_backend != "pyscf":
                raise ValueError(
                    "the ecc and ncc backends (cc_backend keyword argument) are "
                    "designed for closed-shell systems only"
                )
            logger.warning(
                "Warning: All open-shell CC calculations with the pyscf backend "
                "estimate the unrestricted CC property on the basis of a ROHF "
                "reference function instead of the fully restricted CC property."
            )
        if mbe.base_method == "ccsdt" and mbe.cc_backend == "pyscf":
            raise ValueError(
                "ccsdt (base_method keyword argument) is not available with pyscf "
                "coupled-cluster backend (cc_backend keyword argument)"
            )
        if mbe.base_method in ["ccsdt(q)", "ccsdtq"] and mbe.cc_backend != "ncc":
            raise ValueError(
                "ccsdt(q) and ccsdtq (base_method keyword argument) are not available "
                "with the pyscf and ecc coupled-cluster backends (cc_backend keyword "
                "argument)"
            )

    # hf_guess
    if not isinstance(mbe.hf_guess, bool):
        raise TypeError(
            "hf initial guess for fci calculations (hf_guess keyword argument) must be "
            "a bool"
        )
    if mbe.method != "fci" and not mbe.hf_guess:
        raise ValueError(
            "non-hf initial guess (hf_guess keyword argument) only valid for fci calcs "
            "(method keyword argument)"
        )
    if (
        isinstance(mbe.nelec, np.ndarray)
        and isinstance(mbe.fci_state_sym, (str, int))
        and isinstance(mbe.fci_state_root, int)
        and not hasattr(mbe, "fci_state_weights")
        and isinstance(mbe.orbsym, np.ndarray)
    ):
        hf_wfnsym = ground_state_sym(mbe.orbsym, mbe.nelec, mbe.point_group)
        if mbe.method == "fci" and mbe.hf_guess and mbe.fci_state_sym != hf_wfnsym:
            raise ValueError(
                "illegal choice of fci state wavefunction symmetry (fci_state_sym "
                "keyword argument) when enforcing hf initial guess (hf_guess keyword "
                "argument) because fci_state_sym does not equal hf state symmetry"
            )

    # dryrun
    if not isinstance(mbe.dryrun, bool):
        raise TypeError("dryrun option (dryrun keyword argument) must be a bool")
    if mbe.dryrun:
        logger.warning(
            "Warning: Calculation is a dryrun and will skip actual CASCI calculations."
        )

    # exclude single excitations
    if mbe.target in ["rdm12", "genfock"]:
        if not isinstance(mbe.no_singles, bool):
            raise TypeError(
                "excluding single excitations (no_singles keyword argument) must be a "
                "bool"
            )

    # optional integrals for (transition) dipole moment
    if mbe.target in ["dipole", "trans"]:
        if not isinstance(mbe.dipole_ints, np.ndarray):
            raise TypeError(
                "dipole integrals (dipole_ints keyword argument) must be a np.ndarray"
            )
        if mbe.dipole_ints.shape != (3, mbe.norb, mbe.norb):
            raise ValueError(
                "dipole integrals (dipole_ints keyword argument) must have shape "
                "(3, norb, norb)"
            )

    # optional parameters for generalized Fock matrix
    if mbe.target == "genfock":
        if not (hasattr(mbe, "full_norb") and isinstance(mbe.full_norb, int)):
            raise TypeError(
                "number of orbitals in the full system (full_norb keyword argument) "
                "must be an int"
            )
        if mbe.full_norb <= 0:
            raise ValueError(
                "number of orbitals in the full system (full_norb keyword argument) "
                "must be > 0"
            )
        if not (hasattr(mbe, "full_nocc") and isinstance(mbe.full_nocc, int)):
            raise TypeError(
                "number of occupied orbitals in the full system (full_nocc keyword "
                "argument) must be an int"
            )
        if mbe.full_nocc < 0:
            raise ValueError(
                "number of occupied orbitals in the full system (full_nocc keyword "
                "argument) must be >= 0"
            )
        if not (hasattr(mbe, "inact_fock") and isinstance(mbe.inact_fock, np.ndarray)):
            raise TypeError(
                "inactive Fock matrix (inact_fock keyword argument) must be a "
                "np.ndarray"
            )
        if mbe.inact_fock.shape != (mbe.full_norb, mbe.full_nocc + mbe.norb):
            raise ValueError(
                "inactive Fock matrix (inact_fock keyword argument) must have shape "
                "(full_norb, full_nocc + norb)"
            )
        if not (hasattr(mbe, "eri_goaa") and isinstance(mbe.eri_goaa, np.ndarray)):
            raise TypeError(
                "general-occupied-active-active electron repulsion integral (eri_goaa "
                "keyword argument) must be a np.ndarray"
            )
        if mbe.eri_goaa.shape != (mbe.full_norb, mbe.full_nocc, mbe.norb, mbe.norb):
            raise ValueError(
                "general-occupied-active-active electron repulsion integral (eri_goaa "
                "keyword argument) must have shape "
                "(mbe.full_norb, mbe.full_nocc, mbe.norb, mbe.norb)"
            )
        if not (hasattr(mbe, "eri_gaao") and isinstance(mbe.eri_gaao, np.ndarray)):
            raise TypeError(
                "general-active-active-occupied electron repulsion integral (eri_gaao "
                "keyword argument) must be a np.ndarray"
            )
        if mbe.eri_gaao.shape != (mbe.full_norb, mbe.norb, mbe.norb, mbe.full_nocc):
            raise ValueError(
                "general-active-active-occupied electron repulsion integral (eri_gaao "
                "keyword argument) must have shape "
                "(mbe.full_norb, mbe.norb, mbe.norb, mbe.full_nocc)"
            )
        if not (hasattr(mbe, "eri_gaaa") and isinstance(mbe.eri_gaaa, np.ndarray)):
            raise TypeError(
                "general-active-active-active electron repulsion integral (eri_gaaa "
                "keyword argument) must be a np.ndarray"
            )
        if mbe.eri_gaaa.shape != (mbe.full_norb, mbe.norb, mbe.norb, mbe.norb):
            raise ValueError(
                "general-active-active-active electron repulsion integral (eri_gaaa "
                "keyword argument) must have shape "
                "(mbe.full_norb, mbe.norb, mbe.norb, mbe.norb)"
            )


def calc_setup(mbe: MBE) -> MBE:
    """
    this function writes the restart files and distributes all the information between
    processes
    """
    # bcast keywords
    mbe = kw_dist(mbe)

    # bcast system quantities
    mbe = system_dist(mbe)

    # configure logging
    logger_config(mbe.verbose)

    return mbe


def restart_write_kw(mbe: MBE) -> None:
    """
    this function writes the keyword restart file
    """
    # define keywords
    keywords = [
        "method",
        "target",
        "point_group",
        "fci_state_sym",
        "fci_state_root",
        "orb_type",
        "ref_thres",
        "base_method",
        "screen_type",
        "screen_start",
        "screen_perc",
        "screen_thres",
        "screen_func",
        "max_order",
        "rst",
        "rst_freq",
        "fci_backend",
        "cc_backend",
        "hf_guess",
        "verbose",
        "dryrun",
        "no_singles",
    ]

    # put keyword attributes that exist into dictionary
    kw_dict = {}
    for kw in keywords:
        if hasattr(mbe, kw):
            kw_dict[kw] = getattr(mbe, kw)

    # write keywords
    with open(os.path.join(RST, "keywords.rst"), "w") as f:
        dump(kw_dict, f)


def restart_read_kw(mbe: MBE) -> MBE:
    """
    this function reads the keyword restart file
    """
    # read keywords
    with open(os.path.join(RST, "keywords.rst"), "r") as f:
        keywords = load(f)

    # set keywords as MBE attributes
    for key, val in keywords.items():
        setattr(mbe, key, val)

    return mbe


def restart_write_system(mbe: MBE) -> None:
    """
    this function writes all system quantities restart files
    """
    # define system quantities
    system = [
        "norb",
        "nelec",
        "hcore",
        "eri",
        "ref_space",
        "exp_space",
        "base_prop",
        "dipole_ints",
        "full_norb",
        "full_nocc",
        "inact_fock",
        "eri_goaa",
        "eri_gaao",
        "eri_gaaa",
    ]

    # deal with localized orbital symmetry
    if isinstance(mbe.orbsym, np.ndarray):
        system.append("orbsym")
    else:
        with open(os.path.join(RST, "orbsym_local.rst"), "w") as f:
            dump(mbe.orbsym, f)

    # put keyword attributes that exist into dictionary
    system_dict = {}
    for attr in system:
        if hasattr(mbe, attr):
            attr_value = getattr(mbe, attr)
            if not isinstance(attr_value, (tuple, list)):
                system_dict[attr] = attr_value
            else:
                for idx in range(len(attr_value)):
                    system_dict[attr + "_" + str(idx)] = attr_value[idx]

    # write system quantities
    np.savez(os.path.join(RST, "system"), **system_dict)


def restart_read_system(mbe: MBE) -> MBE:
    """
    this function reads all system quantities restart files
    """
    # read system quantities
    system_npz = np.load(os.path.join(RST, "system.npz"))

    # create system dictionary
    system = {}
    for file in system_npz.files:
        system[file] = system_npz[file]

    # close npz object
    system_npz.close()

    # revert to scalars
    for key, value in system.items():
        if value.ndim == 0:
            system[key] = value.item()

    # revert to tuple or list
    tuple_keys = set()
    for key in system.keys():
        key_split = key.split("_")
        if key_split[-1].isdigit():
            tuple_keys.add("_".join(key_split[:-1]))
    for key in tuple_keys:
        n = 0
        attr_list = []
        attr_value = system.pop(key + "_" + str(n), None)
        while attr_value is not None:
            attr_list.append(attr_value)
            n += 1
            attr_value = system.pop(key + "_" + str(n), None)
        if key == "exp_space":
            system[key] = list(attr_list)
        else:
            system[key] = tuple(attr_list)

    # set system quantities as MBE attributes
    for key, val in system.items():
        setattr(mbe, key, val)

    # load localized orbital symmetry
    if not hasattr(mbe, "orbsym"):
        with open(os.path.join(RST, "orbsym_local.rst"), "r") as f:
            mbe.orbsym = load(f)

    return mbe


def restart_write_clustering(
    max_cluster_size: int,
    symm_eqv_sets: Optional[List[List[List[int]]]],
    orb_pairs: np.ndarray,
) -> None:
    """
    this function writes the clustering restart files
    """
    # write files
    np.save(os.path.join(RST, "orb_pairs.npy"), orb_pairs)
    with open(os.path.join(RST, "clustering.rst"), "w") as f:
        dump([max_cluster_size, symm_eqv_sets], f)


def restart_read_clustering() -> (
    Tuple[int, Optional[List[List[List[int]]]], np.ndarray]
):
    """
    this function reads the clustering restart files
    """
    # read files
    orb_pairs = np.load(os.path.join(RST, "orb_pairs.npy"))
    with open(os.path.join(RST, "clustering.rst"), "r") as f:
        max_cluster_size, symm_eqv_sets = load(f)

    return max_cluster_size, symm_eqv_sets, orb_pairs


def ref_space_update(
    tup_sq_overlaps: TupSqOverlapType,
    ref_space: np.ndarray,
    exp_space: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function adds to orbitals to the reference space
    """
    if len(tup_sq_overlaps["overlap"]) == 1:
        # add all orbitals from tuple
        add_orbs = tup_sq_overlaps["tup"][0]

    else:
        # get unique orbitals and counts in tuples with minimum squared overlap
        tup_concat = np.concatenate(tup_sq_overlaps["tup"])
        unique_min, counts_min = np.unique(tup_concat, return_counts=True)

        # add overlapping orbitals between tuples with minimum squared overlap values
        add_orbs = np.atleast_1d(
            unique_min[np.nonzero(counts_min == np.max(counts_min))]
        )

    # add orbitals to reference space
    ref_space = np.append(ref_space, add_orbs)
    ref_space.sort()

    # determine new expansion space
    new_exp_space = []
    for cluster in exp_space:
        add_cluster = np.isin(cluster, add_orbs)
        if np.any(add_cluster):
            if not np.all(add_cluster):
                raise NotImplementedError("Partial cluster is added to reference space")
        else:
            new_exp_space.append(cluster)

    # log results
    logger.info(ref_space_results(add_orbs, ref_space))

    return ref_space, new_exp_space
