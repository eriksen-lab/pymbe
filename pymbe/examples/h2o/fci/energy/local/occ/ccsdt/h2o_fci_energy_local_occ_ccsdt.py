import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, base, ref_mo, ints


def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.Mole()
        mol.build(
            verbose=0,
            output=None,
            atom="""
            O  0.00000000  0.00000000  0.10840502
            H -0.75390364  0.00000000 -0.47943227
            H  0.75390364  0.00000000 -0.47943227
            """,
            basis="631g",
            symmetry="c2v",
        )

        # frozen core
        ncore = 1

        # hf calculation
        nocc, nvirt, norb, hf_object, hf_energy, _, occup, orbsym, mo_coeff = hf(mol)

        # base model
        base_energy = base(
            "ccsdt",
            mol,
            hf_object,
            mo_coeff,
            occup,
            orbsym,
            norb,
            ncore,
            nocc,
            cc_backend="ecc",
        )

        # pipek-mezey localized orbitals
        mo_coeff, orbsym = ref_mo(
            "local", mol, hf_object, mo_coeff, occup, orbsym, norb, ncore, nocc, nvirt
        )

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # create mbe object
        mbe = MBE(
            method="fci",
            cc_backend="ecc",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            hf_prop=hf_energy,
            occup=occup,
            orb_type="local",
            hcore=hcore,
            vhf=vhf,
            eri=eri,
            ref_space=ref_space,
            base_method="ccsdt",
            base_prop=base_energy,
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
