import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints, ref_prop, linear_orbsym


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.Mole()
        mol.build(
            verbose=0,
            output=None,
            atom="""
            C  0.  0.  .7
            C  0.  0. -.7
            """,
            basis="631g",
            symmetry="d2h",
        )

        # frozen core
        ncore = 2

        # hf calculation
        _, hf_prop, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff)

        # reference property
        ref_energy = ref_prop(
            mol, hcore, eri, ref_space, orbsym=orbsym, hf_prop=hf_prop, vhf=vhf
        )

        # pi_pruning
        orbsym_linear = linear_orbsym(mol, mo_coeff)

        # create mbe object
        mbe = MBE(
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
            pi_prune=True,
            orbsym_linear=orbsym_linear,
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
