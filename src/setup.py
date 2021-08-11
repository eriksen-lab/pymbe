#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import numpy as np
from json import load, dump
from mpi4py import MPI
try:
    from pyscf import symm, scf
except ImportError:
    sys.stderr.write('\nImportError : pyscf module not found\n\n')
from typing import Tuple

from parallel import MPICls, mol_dist, calc_dist, fund_dist, prop_dist, mpi_bcast
from system import MolCls, set_system, translate_system, sanity_check as system_sanity_check
from calculation import CalcCls, set_calc, sanity_check as calculation_sanity_check
from expansion import ExpCls
from kernel import gauge_origin, hf, ref_mo, ints, dipole_ints, base, ref_prop
from tools import pi_space, natural_keys, assertion


# restart folder
RST = os.getcwd()+'/rst'


def main() -> Tuple[MPICls, MolCls, CalcCls, ExpCls]:
        """
        this function initializes and broadcasts mpi, mol, calc, and exp objects
        """
        # mpi object
        mpi = MPICls()

        # mol object
        mol = _mol(mpi)

        # calc object
        calc = _calc(mpi, mol)

        # exp object
        mol, calc, exp = _exp(mpi, mol, calc)

        return mpi, mol, calc, exp


def _mol(mpi: MPICls) -> MolCls:
        """
        this function initializes a mol object
        """
        # mol object
        mol = MolCls()
        mol.defaults()

        # input handling
        if mpi.global_master:

            # read input
            mol = set_system(mol)

            # translate input
            mol = translate_system(mol)

            # sanity check
            system_sanity_check(mol)

        # bcast info from master to slaves
        mol = mol_dist(mpi, mol)

        # make pyscf mol object
        mol.make()

        return mol


def _calc(mpi: MPICls, mol: MolCls) -> CalcCls:
        """
        this function initializes a calc object
        """
        # calc object
        calc = CalcCls(mol.ncore, mol.nelectron, mol.groupname)

        # input handling
        if mpi.global_master:

            # read input
            calc = set_calc(calc)

            # set target
            calc.target_mbe = [x for x in calc.target.keys() if calc.target[x]][0]

            # hf symmetry
            if calc.hf_ref['symmetry'] is None:
                calc.hf_ref['symmetry'] = mol.symmetry

            # sanity check
            calculation_sanity_check(calc, mol.spin, mol.atom, mol.groupname)

            # restart folder and logical
            if not os.path.isdir(RST):
                os.mkdir(RST)
                calc.restart = False
            else:
                calc.restart = True

        # bcast info from master to slaves
        calc = calc_dist(mpi, calc)

        return calc


def _exp(mpi: MPICls, mol: MolCls, calc: CalcCls) -> Tuple[MolCls, CalcCls, ExpCls]:
        """
        this function initializes an exp object
        """
        # nuclear repulsion energy
        mol.e_nuc = np.asscalar(mol.energy_nuc()) if mol.atom else 0.

        # dipole gauge origin
        if mol.atom:
            mol.gauge_origin = gauge_origin(mol)
        else:
            mol.gauge_origin = None

        if mpi.global_master:

            if calc.restart:

                # read fundamental info
                mol, calc = restart_read_fund(mol, calc)

                # read properties
                mol, calc = restart_read_prop(mol, calc)

            else:

                # hf calculation
                mol.nocc, mol.nvirt, mol.norb, calc.hf, \
                    calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
                    calc.occup, calc.orbsym, calc.mo_coeff = hf(mol, calc.hf_ref)

                # reference and expansion spaces and mo coefficients
                calc.mo_coeff, calc.nelec, calc.ref_space = ref_mo(mol, calc.mo_coeff, calc.occup, calc.orbsym, \
                                                                   calc.orbs, calc.ref, calc.model, \
                                                                   calc.extra['pi_prune'], calc.hf)

        # bcast fundamental info
        mol, calc = fund_dist(mpi, mol, calc)

        # get handles to all integral windows
        mol.hcore, mol.vhf, mol.eri = ints(mol, calc.mo_coeff, mpi.global_master, mpi.local_master, \
                                           mpi.global_comm, mpi.local_comm, mpi.master_comm, mpi.num_masters)

        # get dipole integrals
        if mol.atom:
            mol.dipole_ints = dipole_ints(mol, calc.mo_coeff)
        else:
            mol.dipole_ints = None

        # write fundamental info
        if not calc.restart and mpi.global_master and calc.misc['rst']:
            restart_write_fund(mol, calc)

        # pyscf hf object not needed anymore
        if mpi.global_master and not calc.restart:
            del calc.hf

        if mpi.global_master:

            # base energy
            if calc.base['method'] is not None:
                calc.prop['base']['energy'], \
                    calc.prop['base']['dipole'] = base(mol, calc.orbs['type'], calc.occup, calc.orbsym, \
                                                       calc.target_mbe, calc.base['method'], \
                                                       calc.model['cc_backend'], calc.prop['hf']['dipole'])
            else:
                calc.prop['base']['energy'] = 0.
                calc.prop['base']['dipole'] = np.zeros(3, dtype=np.float64)

            # reference space properties
            calc.prop['ref'][calc.target_mbe] = ref_prop(mol, calc.occup, calc.target_mbe, \
                                                         calc.orbsym, calc.model['hf_guess'], \
                                                         calc.ref_space, calc.model, calc.orbs['type'], \
                                                         calc.state, calc.prop['hf']['energy'], \
                                                         calc.prop['hf']['dipole'], calc.base['method'])

        # bcast properties
        calc = prop_dist(mpi, calc)

        # write properties
        if not calc.restart and mpi.global_master and calc.misc['rst']:
            restart_write_prop(mol, calc)

        # exp object
        exp = ExpCls(mol, calc)

        # possible restart
        if calc.restart:
            exp.start_order = restart_main(mpi, calc, exp)
        else:
            exp.start_order = exp.min_order

        # pi-orbital space
        if calc.extra['pi_prune']:

            # recast mol in parent point group (dooh/coov) - make pi-space based on those symmetries
            mol_parent = mol.copy()
            parent_group = 'Dooh' if mol.groupname == 'D2h' else 'Coov'
            mol_parent = mol_parent.build(0, 0, symmetry=parent_group)

            orbsym_parent = symm.label_orb_symm(mol_parent, mol_parent.irrep_id, \
                                                mol_parent.symm_orb, calc.mo_coeff)

            # pi-space
            exp.pi_orbs, exp.pi_hashes = pi_space(parent_group, orbsym_parent, exp.exp_space[0])

        return mol, calc, exp


def restart_main(mpi: MPICls, calc: CalcCls, exp: ExpCls) -> int:
        """
        this function reads in all expansion restart files and returns the start order
        """
        # list sorted filenames in files list
        if mpi.global_master:
            files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
            files.sort(key = natural_keys)

        # distribute filenames
        if mpi.global_master:
            mpi.global_comm.bcast(files, root=0)
        else:
            files = mpi.global_comm.bcast(None, root=0)

        # loop over n_tuples files
        if mpi.global_master:
            for i in range(len(files)):
                if 'mbe_n_tuples' in files[i]:
                    if 'theo' in files[i]:
                        exp.n_tuples['theo'].append(np.load(os.path.join(RST, files[i])).tolist())
                    if 'inc' in files[i]:
                        exp.n_tuples['inc'].append(np.load(os.path.join(RST, files[i])).tolist())
            mpi.global_comm.bcast(exp.n_tuples, root=0)
        else:
            exp.n_tuples = mpi.global_comm.bcast(None, root=0)

        # loop over all other files
        for i in range(len(files)):

            # read hashes
            if 'mbe_hashes' in files[i]:
                n_tuples = exp.n_tuples['inc'][len(exp.prop[calc.target_mbe]['hashes'])]
                exp.prop[calc.target_mbe]['hashes'].append(MPI.Win.Allocate_shared(8 * n_tuples if mpi.local_master else 0, \
                                                                                   8, comm=mpi.local_comm))
                buf = exp.prop[calc.target_mbe]['hashes'][-1].Shared_query(0)[0] # type: ignore
                hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tuples,))
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif 'mbe_inc' in files[i]:
                n_tuples = exp.n_tuples['inc'][len(exp.prop[calc.target_mbe]['inc'])]
                if mpi.local_master:
                    if calc.target_mbe in ['energy', 'excitation']:
                        exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(8 * n_tuples, 8, comm=mpi.local_comm))
                    elif calc.target_mbe in ['dipole', 'trans']:
                        exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(8 * n_tuples * 3, 8, comm=mpi.local_comm))
                else:
                    exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm))
                buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
                if calc.target_mbe in ['energy', 'excitation']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tuples,))
                elif calc.target_mbe in ['dipole', 'trans']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tuples, 3))
                if mpi.global_master:
                    inc[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[:] = mpi_bcast(mpi.master_comm, inc)
                mpi.local_comm.Barrier()

            if mpi.global_master:

                # read expansion spaces
                if 'exp_space' in files[i]:
                    exp.exp_space.append(np.load(os.path.join(RST, files[i])))

                # read total properties
                elif 'mbe_screen' in files[i]:
                    exp.screen = np.load(os.path.join(RST, files[i]))

                # read total properties
                elif 'mbe_tot' in files[i]:
                    exp.prop[calc.target_mbe]['tot'].append(np.load(os.path.join(RST, files[i])))

                # read ndets statistics
                elif 'mbe_mean_ndets' in files[i]:
                    exp.mean_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_ndets' in files[i]:
                    exp.min_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_ndets' in files[i]:
                    exp.max_ndets.append(np.load(os.path.join(RST, files[i])))

                # read inc statistics
                elif 'mbe_mean_inc' in files[i]:
                    exp.mean_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_inc' in files[i]:
                    exp.min_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_inc' in files[i]:
                    exp.max_inc.append(np.load(os.path.join(RST, files[i])))

                # read timings
                elif 'mbe_time_mbe' in files[i]:
                    exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
                elif 'mbe_time_purge' in files[i]:
                    exp.time['purge'].append(np.load(os.path.join(RST, files[i])).tolist())

        # bcast exp_space and screen
        if mpi.global_master:
            mpi.global_comm.bcast(exp.exp_space, root=0)
            mpi.global_comm.bcast(exp.screen, root=0)
        else:
            exp.exp_space = mpi.global_comm.bcast(None, root=0)
            exp.screen = mpi.global_comm.bcast(None, root=0)

        # mpi barrier
        mpi.global_comm.Barrier()

        return exp.min_order + len(exp.prop[calc.target_mbe]['tot'])


def restart_write_fund(mol: MolCls, calc: CalcCls) -> None:
        """
        this function writes all fundamental info restart files
        """
        # write dimensions
        dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'norb': mol.norb, 'nelec': calc.nelec}
        with open(os.path.join(RST, 'dims.rst'), 'w') as f:
            dump(dims, f)

        # write reference & expansion spaces
        np.save(os.path.join(RST, 'ref_space'), calc.ref_space)

        # occupation
        np.save(os.path.join(RST, 'occup'), calc.occup)

        # write orbital coefficients
        np.save(os.path.join(RST, 'mo_coeff'), calc.mo_coeff)

        # write fock, coulomb and exchange matrixes
        fao = calc.hf.get_fock()
        fmo = calc.mo_coeff.T @ fao @ calc.mo_coeff
        mocc = calc.mo_coeff[:,calc.occup>0]
        dm = np.dot(mocc*calc.occup[calc.occup>0], mocc.T)
        jmo, kmo = calc.hf.get_jk(mol, dm)
        np.save(os.path.join(RST, 'fock'), fmo)
        np.save(os.path.join(RST, 'coulomb'), jmo)
        np.save(os.path.join(RST, 'exchange'), kmo)


def restart_read_fund(mol: MolCls, calc: CalcCls) -> Tuple[MolCls, CalcCls]:
        """
        this function reads all fundamental info restart files
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key = natural_keys)

        # loop over files
        for i in range(len(files)):

            # read dimensions
            if 'dims' in files[i]:
                with open(os.path.join(RST, files[i]), 'r') as f:
                    dims = load(f)
                mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']
                mol.norb = dims['norb']; calc.nelec = dims['nelec']

            # read reference space
            elif 'ref_space' in files[i]:
                calc.ref_space = np.load(os.path.join(RST, files[i]))

            # read occupation
            elif 'occup' in files[i]:
                calc.occup = np.load(os.path.join(RST, files[i]))

            # read orbital coefficients
            elif 'mo_coeff' in files[i]:
                calc.mo_coeff = np.load(os.path.join(RST, files[i]))
                if mol.symmetry:
                    calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
                else:
                    calc.orbsym = np.zeros(mol.norb, dtype=np.int64)

        return mol, calc


def restart_write_prop(mol: MolCls, calc: CalcCls) -> None:
        """
        this function writes all property restart files
        """
        # write hf, reference, and base properties
        energies = {'hf': calc.prop['hf']['energy']}
        dipoles = {'hf': calc.prop['hf']['dipole'].tolist()} # type: ignore

        if calc.target_mbe == 'energy':

            energies['base'] = calc.prop['base']['energy']
            energies['ref'] = calc.prop['ref']['energy']

        with open(os.path.join(RST, 'energies.rst'), 'w') as f:
            dump(energies, f)

        if calc.target_mbe == 'excitation':

            excitations = {'ref': calc.prop['ref']['excitation']}
            with open(os.path.join(RST, 'excitations.rst'), 'w') as f:
                dump(excitations, f)

        if calc.target_mbe == 'dipole':

            dipoles['base'] = calc.prop['base']['dipole'].tolist() # type: ignore
            dipoles['ref'] = calc.prop['ref']['dipole'].tolist() # type: ignore

        with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
            dump(dipoles, f)

        if calc.target_mbe == 'trans':

            transitions = {'ref': calc.prop['ref']['trans'].tolist()} # type: ignore
            with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
                dump(transitions, f)


def restart_read_prop(mol: MolCls, calc: CalcCls) -> Tuple[MolCls, CalcCls]:
        """
        this function reads all property restart files
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key = natural_keys)

        # loop over files
        for i in range(len(files)):

            # read hf and base properties
            if 'energies' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    energies = load(f)
                calc.prop['hf']['energy'] = energies['hf']

                if calc.target_mbe == 'energy':
                    calc.prop['base']['energy'] = energies['base']
                    calc.prop['ref']['energy'] = energies['ref']

            if 'excitations' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    excitations = load(f)
                calc.prop['ref']['excitation'] = excitations['ref']

            if 'dipoles' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    dipoles = load(f)
                calc.prop['hf']['dipole'] = np.asarray(dipoles['hf'])

                if calc.target_mbe == 'dipole':
                    calc.prop['base']['dipole'] = np.asarray(dipoles['base'])
                    calc.prop['ref']['dipole'] = np.asarray(dipoles['ref'])

            if 'transitions' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    transitions = load(f)
                calc.prop['ref']['trans'] = np.asarray(transitions['ref'])

        return mol, calc


def settings() -> None:
        """
        this function sets and asserts some general settings
        """
        # only run with python3+
        assertion(3 <= sys.version_info[0], 'PyMBE only runs under python3+')

        # mute scf checkpoint files
        scf.hf.MUTE_CHKFILE = True

        # PYTHONHASHSEED = 0
        pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
        assertion(int(pythonhashseed) == 0, \
                  'environment variable PYTHONHASHSEED must be set to zero')


