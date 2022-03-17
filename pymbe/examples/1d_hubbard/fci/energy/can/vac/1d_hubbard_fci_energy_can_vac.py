import os
from math import floor
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.M(verbose=0)

        # hubbard hamiltonian
        u = 2.0
        n = 1.0
        matrix = (1, 10)
        pbc = True

        # number of orbitals
        mol.nao = np.array(matrix[1], dtype=np.int64)

        # number of electrons
        mol.nelectron = floor(matrix[0] * matrix[1] * n)

        # hf calculation
        _, hf_prop, orbsym, mo_coeff = hf(mol, u=u, matrix=matrix, pbc=pbc)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff, u=u, matrix=matrix, pbc=pbc)

        # create mbe object
        mbe = MBE(
            mol=mol,
            hf_prop=hf_prop,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            vhf=vhf,
            rst=rst,
        )

        # perform calculation
        energy = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        energy = mbe.kernel()

    return energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
