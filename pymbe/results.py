#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from pyscf import gto, symm
from typing import TYPE_CHECKING

try:
    import matplotlib

    PLT_FOUND = True
except (ImportError, OSError):
    pass
    PLT_FOUND = False
if PLT_FOUND:
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    try:
        import seaborn as sns

        SNS_FOUND = True
    except (ImportError, OSError):
        pass
        SNS_FOUND = False

from pymbe.tools import intervals, time_str, get_nelec

if TYPE_CHECKING:

    from typing import Dict, Union, List, Optional

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


# results parameters
DIVIDER = f"{('-' * 137):^143}"


def atom_prt(mol: gto.Mole) -> str:
    """
    this function returns the molecular geometry
    """
    # print atom
    string: str = DIVIDER[:39] + "\n"
    string += f"{'geometry':^45}\n"
    string += DIVIDER[:39] + "\n"
    molecule = gto.tostring(mol).split("\n")
    for i in range(len(molecule)):
        atom = molecule[i].split()
        for j in range(1, 4):
            atom[j] = float(atom[j])
        string += (
            f"   {atom[0]:<3s} {atom[1]:>10.5f} {atom[2]:>10.5f} {atom[3]:>10.5f}\n"
        )
    string += DIVIDER[:39] + "\n"
    return string


def _model(method: str) -> str:
    """
    this function returns the expansion model
    """
    string = f"{method.upper()}"

    return string


def _state(spin: int, root: int) -> str:
    """
    this function returns the state of interest
    """
    string = f"{root}"
    if spin == 0:
        string += " (singlet)"
    elif spin == 1:
        string += " (doublet)"
    elif spin == 2:
        string += " (triplet)"
    elif spin == 3:
        string += " (quartet)"
    elif spin == 4:
        string += " (quintet)"
    else:
        string += f" ({spin + 1})"
    return string


def _base(base_method: Optional[str]) -> str:
    """
    this function returns the base model
    """
    if base_method is None:
        return "none"
    else:
        return base_method.upper()


def _system(ref_space: np.ndarray, exp_space: np.ndarray, occup: np.ndarray) -> str:
    """
    this function returns the system size
    """
    return (
        f"{get_nelec(occup, np.concatenate(ref_space + exp_space))} e in "
        f"{ref_space.size + exp_space.size} o"
    )


def _solver(method: str, spin: int) -> str:
    """
    this function returns the chosen fci solver
    """
    if method != "fci":
        return "none"
    else:
        if spin == 0:
            return "PySCF (spin0)"
        else:
            return "PySCF (spin1)"


def _active_space(occup: np.ndarray, ref_space: np.ndarray) -> str:
    """
    this function returns the active space
    """
    act_nelec = get_nelec(occup, ref_space)
    string = f"{act_nelec[0] + act_nelec[1]} e, {ref_space.size} o"
    return string


def _active_orbs(ref_space: np.ndarray) -> str:
    """
    this function returns the orbitals of the active space
    """
    if ref_space.size == 0:
        return "none"

    # init string
    string = "["
    # divide ref_space into intervals
    ref_space_ints = [i for i in intervals(ref_space)]

    for idx, i in enumerate(ref_space_ints):
        elms = f"{i[0]}-{i[1]}" if len(i) > 1 else f"{i[0]}"
        string += f"{elms}," if idx < len(ref_space_ints) - 1 else f"{elms}"
    string += "]"

    return string


def _orbs(orb_type: str) -> str:
    """
    this function returns the choice of orbitals
    """
    if orb_type == "can":
        return "canonical"
    elif orb_type == "ccsd":
        return "CCSD NOs"
    elif orb_type == "ccsd(t)":
        return "CCSD(T) NOs"
    elif orb_type == "local":
        return "pipek-mezey"
    elif orb_type == "casscf":
        return "casscf"
    else:
        raise NotImplementedError("unknown orbital basis")


def _mpi(num_masters: int, global_size: int) -> str:
    """
    this function returns the mpi information
    """
    return f"{num_masters} & {global_size - num_masters}"


def _point_group(point_group: str) -> str:
    """
    this function returns the point group symmetry
    """
    return point_group


def _symm(method: str, point_group: str, fci_state_sym: int, pi_prune: bool) -> str:
    """
    this function returns the symmetry of the wavefunction in the computational point
    group
    """
    if method == "fci":
        string = (
            symm.addons.irrep_id2name(point_group, fci_state_sym)
            + "("
            + point_group
            + ")"
        )
        if pi_prune:
            string += " (pi)"
        return string
    else:
        return "unknown"


def _time(time: Dict[str, List[float]], comp: str, idx: int) -> str:
    """
    this function returns the final timings in (HHH : MM : SS) format
    """
    # init time
    if comp in ["mbe", "purge"]:
        req_time = time[comp][idx]
    elif comp == "sum":
        req_time = time["mbe"][idx] + time["purge"][idx]
    elif comp in ["tot_mbe", "tot_purge"]:
        req_time = np.sum(time[comp[4:]])
    elif comp == "tot_sum":
        req_time = np.sum(time["mbe"]) + np.sum(time["purge"])
    return time_str(req_time)


def summary_prt(
    mpi: MPICls,
    exp: ExpCls,
    hf_prop: Union[float, np.floating],
    base_prop: Union[float, np.floating],
    mbe_tot_prop: Union[float, np.floating],
) -> str:
    """
    this function returns the summary table
    """
    string: str = DIVIDER + "\n"
    string += (
        f"{'':3}{'molecular information':^45}{'|':1}"
        f"{'expansion information':^45}{'|':1}"
        f"{'calculation information':^45}\n"
    )

    string += DIVIDER + "\n"
    string += (
        f"{'':5}{'system size':<24}{'=':1}{'':2}"
        f"{_system(exp.ref_space, exp.exp_space[0], exp.occup):<16s}{'|':1}{'':2}"
        f"{'expansion model':<24}{'=':1}{'':2}"
        f"{_model(exp.method):<16s}{'|':1}{'':2}"
        f"{'mpi masters & slaves':<24}{'=':1}{'':2}"
        f"{_mpi(mpi.num_masters, mpi.global_size):<16s}\n"
    )
    string += (
        f"{'':5}{'state (mult.)':<24}{'=':1}{'':2}"
        f"{_state(exp.spin, exp.fci_state_root):<16s}{'|':1}{'':2}"
        f"{'reference space':<24}{'=':1}{'':2}"
        f"{_active_space(exp.occup, exp.ref_space):<16s}{'|':1}{'':2}"
        f"{('Hartree-Fock ' + exp.target):<24}{'=':1}{'':2}"
        f"{hf_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'orbitals':<24}{'=':1}{'':2}"
        f"{_orbs(exp.orb_type):<16s}{'|':1}{'':2}"
        f"{'reference orbs.':<24}{'=':1}{'':2}"
        f"{_active_orbs(exp.ref_space):<16s}{'|':1}{'':2}"
        f"{('base model ' + exp.target):<24}{'=':1}{'':2}"
        f"{base_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'point group':<24}{'=':1}{'':2}"
        f"{_point_group(exp.point_group):<16s}{'|':1}{'':2}"
        f"{'base model':<24}{'=':1}{'':2}"
        f"{_base(exp.base_method):<16s}{'|':1}{'':2}"
        f"{('MBE total ' + exp.target):<24}{'=':1}{'':2}"
        f"{mbe_tot_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'FCI solver':<24}{'=':1}{'':2}"
        f"{_solver(exp.method, exp.spin):<16s}{'|':1}{'':2}"
        f"{'':<24}{'':1}{'':2}"
        f"{'':<16s}{'|':1}{'':2}"
        f"{('total time'):<24}{'=':1}{'':2}"
        f"{_time(exp.time, 'tot_sum', -1):<16s}\n"
    )
    string += (
        f"{'':5}{'wave funct. symmetry':<24}{'=':1}{'':2}"
        f"{_symm(exp.method, exp.point_group, exp.fci_state_sym, exp.pi_prune):<16s}"
        f"{'|':1}{'':2}"
        f"{'':<24}{'':1}{'':2}"
        f"{'':<16s}{'|':1}{'':2}"
        f"{(''):<24}{'':1}{'':2}"
        f"{'':<16s}\n"
    )

    string += DIVIDER + "\n"

    return string


def timings_prt(exp: ExpCls, method: str) -> str:
    """
    this function returns the timings table
    """
    string: str = DIVIDER[:106] + "\n"
    string += f"{f'MBE-{method.upper()} timings':^106}\n"

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}{'MBE':^18}{'|':1}{'purging':^18}{'|':1}"
        f"{'sum':^18}{'|':1}{'calculations':^18}{'|':1}{'in %':^13}\n"
    )

    string += DIVIDER[:106] + "\n"
    for i, j in enumerate(range(exp.min_order, exp.final_order + 1)):
        calc_i = exp.n_tuples["calc"][i]
        rel_i = exp.n_tuples["calc"][i] / exp.n_tuples["theo"][i] * 100.0
        calc_tot = sum(exp.n_tuples["calc"][: i + 1])
        rel_tot = calc_tot / sum(exp.n_tuples["theo"][: i + 1]) * 100.0
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{_time(exp.time, 'mbe', i):>16s}{'':2}{'|':1}"
            f"{_time(exp.time, 'purge', i):>16s}{'':2}{'|':1}"
            f"{_time(exp.time, 'sum', i):>16s}{'':2}{'|':1}"
            f"{calc_i:>16d}{'':2}{'|':1}"
            f"{rel_i:>10.2f}\n"
        )

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'total':^14s}{'|':1}"
        f"{_time(exp.time, 'tot_mbe', -1):>16s}{'':2}{'|':1}"
        f"{_time(exp.time, 'tot_purge', -1):>16s}{'':2}{'|':1}"
        f"{_time(exp.time, 'tot_sum', -1):>16s}{'':2}{'|':1}"
        f"{calc_tot:>16d}{'':2}{'|':1}"
        f"{rel_tot:>10.2f}\n"
    )

    string += DIVIDER[:106] + "\n"

    return string


def results_plt(
    prop: np.ndarray,
    min_order: int,
    final_order: int,
    marker: str,
    color: str,
    label: str,
    ylabel: str,
) -> matplotlib.figure.Figure:
    """
    this function plots the target property
    """
    # check if matplotlib is available
    if not PLT_FOUND:
        raise ModuleNotFoundError("No module named matplotlib")

    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        prop,
        marker=marker,
        linewidth=2,
        mew=1,
        color=color,
        linestyle="-",
        label=label,
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel(ylabel)

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig
