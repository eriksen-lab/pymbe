#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main pymbe program
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
import shutil

from setup import settings, main as setup_main
from driver import master as driver_master, slave as driver_slave
from tools import Logger
from output import OUT, OUT_FILE
from results import RES_FILE, main as results_main
from parallel import mpi_finalize


def main():
        """ main program """
        # general settings
        settings()

        # init mpi, mol, calc, and exp objects
        mpi, mol, calc, exp = setup_main()

        if mpi.global_master:

            # rm out dir if present
            if os.path.isdir(OUT):
                shutil.rmtree(OUT, ignore_errors=True)

            # make out dir
            os.mkdir(OUT)

            # init logger
            sys.stdout = Logger(OUT_FILE)

            # main master driver
            driver_master(mpi, mol, calc, exp)

            # re-init logger
            sys.stdout = Logger(RES_FILE, both=False)

            # print/plot results
            results_main(mpi, mol, calc, exp)

            # finalize mpi
            mpi_finalize(mpi, calc.misc['rst'])

        else:

            # main slave driver
            driver_slave(mpi, mol, calc, exp)


if __name__ == '__main__':
    main()


