import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ref_mo, ints


def mbe_example(rst=True):

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

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # frozen core
        ncore = 1

        # hf calculation
        hf_object, orbsym, mo_coeff = hf(mol)

        # natural orbitals
        mo_coeff, orbsym = ref_mo("ccsd(t)", mol, hf_object, mo_coeff, orbsym, ncore)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # integral calculation
        hcore, eri = ints(mol, mo_coeff)

        # create mbe object
        mbe = MBE(
            mol=mol,
            orbsym=orbsym,
            orb_type="ccsd(t)",
            hcore=hcore,
            eri=eri,
            ref_space=ref_space,
            exp_space=exp_space,
            rst=rst,
        )

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_energy = mbe.kernel()

    # get total energy
    tot_energy = mbe.final_prop(prop_type="total", nuc_prop=mol.energy_nuc().item())

    return tot_energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
