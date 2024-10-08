import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo
from pymbe import MBE


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
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # orbsym
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)

        # expansion space
        exp_space = np.arange(ncore, mol.nao, dtype=np.int64)

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", hf.mo_coeff, hcore_ao, hf.mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, hf.mo_coeff)

        # create mbe object
        mbe = MBE(
            method="ccsd",
            target="rdm12",
            mol=mol,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            exp_space=exp_space,
            rst=rst,
            no_singles=False,
        )

        # perform calculation
        rdm12 = mbe.kernel()

    else:
        # create mbe object
        mbe = MBE()

        # perform calculation
        rdm12 = mbe.kernel()

    return rdm12


if __name__ == "__main__":
    # call example function
    rdm12 = mbe_example()

    # finalize mpi
    MPI.Finalize()
