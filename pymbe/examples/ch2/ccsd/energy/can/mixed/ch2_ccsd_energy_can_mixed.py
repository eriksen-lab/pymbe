import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints, ref_prop


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.Mole()
        mol.build(
            verbose=0,
            output=None,
            atom="""
            C  0.00000  0.00000  0.00000
            H  0.98920  0.42714  0.00000
            H -0.98920  0.42714  0.00000
            """,
            basis="631g",
            symmetry="c2v",
            spin=2,
        )

        # frozen core
        ncore = 1

        # hf calculation
        _, hf_prop, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff)

        # reference property
        ref_energy = ref_prop(
            mol, hcore, eri, orbsym, ref_space, method="ccsd", hf_prop=hf_prop, vhf=vhf
        )

        # create mbe object
        mbe = MBE(
            method="ccsd",
            mol=mol,
            ncore=ncore,
            orbsym=orbsym,
            hf_prop=hf_prop,
            hcore=hcore,
            eri=eri,
            vhf=vhf,
            ref_space=ref_space,
            ref_prop=ref_energy,
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
