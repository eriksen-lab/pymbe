#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
restart module containing all functions related to writing and reading restart files
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import json
import os
import os.path
import shutil
import re
from pyscf import symm


# restart folder
RST = os.getcwd()+'/rst'


def restart():
        """
        this function returns the restart logical

        :return: bool
        """
        if not os.path.isdir(RST):
            os.mkdir(RST)
            return False
        else:
            return True


def rm():
        """
        this function removes the rst directory in case pymbe successfully terminates
        """
        shutil.rmtree(RST)


def main(mpi, calc, exp):
        """
        this function reads in all expansion restart files and returns the start order

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: integer
        """
        if calc.restart:

            # list filenames in files list
            files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

            # sort the list of files
            files.sort(key=_natural_keys)

            # loop over files
            for i in range(len(files)):

                # read tuples
                if 'mbe_tup' in files[i]:
                    exp.tuples = np.load(os.path.join(RST, files[i]))

                # read hashes
                elif 'mbe_hash' in files[i]:
                    exp.hashes.append(np.load(os.path.join(RST, files[i])))

                # read increments
                elif 'mbe_inc' in files[i]:
                    n_tasks = exp.n_tasks[len(exp.prop[calc.target]['inc'])]
                    if mpi.master:
                        exp.prop[calc.target]['inc'].append(MPI.Win.Allocate_shared(8 * n_tasks, 8, comm=mpi.comm))
                    else:
                        exp.prop[calc.target]['inc'].append(MPI.Win.Allocate_shared(0, 8, comm=mpi.comm))
                    buf = exp.prop[calc.target]['inc'][-1].Shared_query(0)[0]
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tasks,))
                    if mpi.master:
                        inc = np.load(os.path.join(RST, files[i]))
#                    exp.prop[calc.target]['inc'].append(np.load(os.path.join(RST, files[i])))

                # read total properties
                elif 'mbe_tot' in files[i]:
                    exp.prop[calc.target]['tot'].append(np.load(os.path.join(RST, files[i])).tolist())

                # read ndets
                elif 'mbe_mean_ndets' in files[i]:
                    exp.mean_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_ndets' in files[i]:
                    exp.min_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_ndets' in files[i]:
                    exp.max_ndets.append(np.load(os.path.join(RST, files[i])))

                # read timings
                elif 'mbe_time_mbe' in files[i]:
                    exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
                elif 'mbe_time_screen' in files[i]:
                    exp.time['screen'].append(np.load(os.path.join(RST, files[i])).tolist())

        return exp.tuples.shape[1]


def mbe_write(order, inc, tot=None, mean_ndets=None, max_ndets=None, min_ndets=None, time_mbe=None):
        """
        this function writes all mbe restart files

        :param order: current mbe order. integer
        :param inc: increments. numpy array of shape (n_tuples,) or (n_tuples, 3) depending on target
        :param tot: total prop. numpy array of shape (order-start_order,) or (order-start_order, 3) depending on target
        :param mean_ndets: mean number of determinants. scalar
        :param max_ndets: max number of determinants. scalar
        :param min_ndets: min number of determinants. scalar
        :param time_mbe: mbe timings. scalar
        :param total: logical controlling whether or not this is final call. bool
        """
        # increments
        np.save(os.path.join(RST, 'mbe_inc_{:}'.format(order)), inc)

        if tot is not None:
            # total properties
            np.save(os.path.join(RST, 'mbe_tot_{:}'.format(order)), tot)

        # write ndets
        if mean_ndets is not None:
            np.save(os.path.join(RST, 'mbe_mean_ndets_'+str(order)), mean_ndets)
        if max_ndets is not None:
            np.save(os.path.join(RST, 'mbe_max_ndets_'+str(order)), max_ndets)
        if min_ndets is not None:
            np.save(os.path.join(RST, 'mbe_min_ndets_'+str(order)), min_ndets)

        # write time
        if time_mbe is not None:
            np.save(os.path.join(RST, 'mbe_time_mbe_'+str(order)), time_mbe)


def screen_write(order, tuples, hashes, time_screen):
        """
        this function writes all screening restart files

        :param order: current mbe order. integer
        :param tuples: tuples. numpy array of shape (n_tuples, order)
        :param hashes: hashes. numpy array of shape (n_tuples,)
        :param time_screen: screening timings. scalar
        """
        # write tuples
        np.save(os.path.join(RST, 'mbe_tup'), tuples)

        # write hashes
        np.save(os.path.join(RST, 'mbe_hash_'+str(order+1)), hashes)

        # write time
        np.save(os.path.join(RST, 'mbe_time_screen_'+str(order)), time_screen)


def write_fund(mol, calc):
        """
        this function writes all fundamental info restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        """
        # write dimensions
        dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'norb': mol.norb, 'nelec': calc.nelec}
        with open(os.path.join(RST, 'dims.rst'), 'w') as f:
            json.dump(dims, f)

        # write hf, reference, and base properties
        if calc.target == 'energy':

            energies = {'e_nuc': mol.e_nuc, \
                        'hf': calc.prop['hf']['energy'], \
                        'base': calc.prop['base']['energy'], \
                        'ref': calc.prop['ref']['energy']}
            with open(os.path.join(RST, 'energies.rst'), 'w') as f:
                json.dump(energies, f)

        elif calc.target == 'excitation':

            excitations = {'ref': calc.prop['ref']['excitation']}
            with open(os.path.join(RST, 'excitations.rst'), 'w') as f:
                json.dump(excitations, f)

        elif calc.target == 'dipole':

            dipoles = {'hf': calc.prop['hf']['dipole'].tolist(), \
                        'ref': calc.prop['ref']['dipole'].tolist()}
            with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
                json.dump(dipoles, f)

        elif calc.target == 'trans':

            transitions = {'ref': calc.prop['ref']['trans'].tolist()}
            with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
                json.dump(transitions, f)

        # write expansion spaces
        np.save(os.path.join(RST, 'ref_space'), calc.ref_space)
        np.save(os.path.join(RST, 'exp_space_tot'), calc.exp_space['tot'])
        np.save(os.path.join(RST, 'exp_space_occ'), calc.exp_space['occ'])
        np.save(os.path.join(RST, 'exp_space_virt'), calc.exp_space['virt'])
        if calc.extra['pi_prune']:
            np.save(os.path.join(RST, 'exp_space_pi_orbs'), calc.exp_space['pi_orbs'])
            np.save(os.path.join(RST, 'exp_space_pi_hashes'), calc.exp_space['pi_hashes'])

        # occupation
        np.save(os.path.join(RST, 'occup'), calc.occup)

        # write orbital energies
        np.save(os.path.join(RST, 'mo_energy'), calc.mo_energy)

        # write orbital coefficients
        np.save(os.path.join(RST, 'mo_coeff'), calc.mo_coeff)


def read_fund(mol, calc):
        """
        this function reads all fundamental info restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: updated mol object,
                 updated calc object
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=_natural_keys)

        # init exp_space
        calc.exp_space = {}

        # loop over files
        for i in range(len(files)):

            # read dimensions
            if 'dims' in files[i]:
                with open(os.path.join(RST, files[i]), 'r') as f:
                    dims = json.load(f)
                mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']
                mol.norb = dims['norb']; calc.nelec = dims['nelec']

            # read hf and base properties
            elif 'energies' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    energies = json.load(f)
                mol.e_nuc = energies['e_nuc']
                calc.prop['hf']['energy'] = energies['hf']
                calc.prop['base']['energy'] = energies['base'] 
                calc.prop['ref']['energy'] = energies['ref']

            elif 'excitations' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    excitations = json.load(f)
                calc.prop['ref']['excitation'] = excitations['ref']

            elif 'dipoles' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    dipoles = json.load(f)
                calc.prop['hf']['dipole'] = np.asarray(dipoles['hf'])
                calc.prop['ref']['dipole'] = np.asarray(dipoles['ref'])

            elif 'transitions' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    transitions = json.load(f)
                calc.prop['ref']['trans'] = np.asarray(transitions['ref'])

            # read expansion spaces
            elif 'ref_space' in files[i]:
                calc.ref_space = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_tot' in files[i]:
                calc.exp_space['tot'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_occ' in files[i]:
                calc.exp_space['occ'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_virt' in files[i]:
                calc.exp_space['virt'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_orbs' in files[i]:
                calc.exp_space['pi_orbs'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_hashes' in files[i]:
                calc.exp_space['pi_hashes'] = np.load(os.path.join(RST, files[i]))

            # read occupation
            elif 'occup' in files[i]:
                calc.occup = np.load(os.path.join(RST, files[i]))

            # read orbital energies
            elif 'mo_energy' in files[i]:
                calc.mo_energy = np.load(os.path.join(RST, files[i]))

            # read orbital coefficients
            elif 'mo_coeff' in files[i]:
                calc.mo_coeff = np.load(os.path.join(RST, files[i]))
                if mol.atom:
                    calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
                else:
                    calc.orbsym = np.zeros(mol.norb, dtype=np.int)

        return mol, calc


def _natural_keys(txt):
        """
        this function return keys to sort a string in human order (as alist.sort(key=natural_keys))
        see: http://nedbatchelder.com/blog/200712/human_sorting.html
        see: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

        :param txt: text. string
        :return: list of keys
        """
        return [_convert(c) for c in re.split('(\d+)', txt)]


def _convert(txt):
        """
        this function converts strings with numbers in them

        :param txt: text. string
        :return: integer or string depending on txt
        """
        return int(txt) if txt.isdigit() else txt


