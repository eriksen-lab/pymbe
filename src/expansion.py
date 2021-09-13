#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Any

from system import MolCls
from calculation import CalcCls


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, mol: MolCls, calc: CalcCls) -> None:
                """
                init expansion attributes
                """
                # set expansion model dict
                self.model = deepcopy(calc.model)

                # init prop dict
                self.prop: Dict[str, Dict[str, Union[List[float], MPI.Win]]] = {str(calc.target_mbe): {'inc': [], 'tot': [], \
                                                                                                       'hashes': []}}

                # init timings and and statistics lists
                self.time: Dict[str, Union[List[float], np.ndarray]] = {'mbe': [], 'purge': []}
                self.mean_inc: Union[List[float], np.ndarray] = []
                self.min_inc: Union[List[float], np.ndarray] = []
                self.max_inc: Union[List[float], np.ndarray] = []
                self.mean_ndets: Union[List[int], np.ndarray] = []
                self.min_ndets: Union[List[int], np.ndarray] = []
                self.max_ndets: Union[List[int], np.ndarray] = []

                # init order
                self.order: int = 0

                # init attributes
                self.start_order: int = 0
                self.final_order: int = 0
                self.screen: np.ndarray = None
                self.screen_orbs: np.ndarray = None
                self.exp_space: List[np.ndarray] = [np.array([i for i in range(mol.ncore, mol.norb) if i not in calc.ref_space], dtype=np.int64)]
                self.n_tuples: Dict[str, List[int]] = {'theo': [], 'inc': []}
                self.pi_orbs: np.ndarray = None
                self.pi_hashes: np.ndarray = None

                # set min_order
                if calc.base['method'] in ['ccsd', 'ccsd(t)', 'ccsdt']:
                    valid_order = 4 - min(2, calc.ref_space[calc.ref_space < mol.nocc].size) - min(2, calc.ref_space[mol.nocc <= calc.ref_space].size)
                elif calc.base['method'] == 'ccsdtq':
                    valid_order = 6 - min(3, calc.ref_space[calc.ref_space < mol.nocc].size) - min(3, calc.ref_space[mol.nocc <= calc.ref_space].size)
                else:
                    valid_order = 2 - calc.ref_space.size
                self.min_order: int = max(1, valid_order)

                # set max_order
                if calc.misc['order'] is not None:
                    self.max_order: int = min(self.exp_space[0].size, calc.misc['order'])
                else:
                    self.max_order: int = self.exp_space[0].size



if __name__ == "__main__":
    import doctest
    doctest.testmod()


