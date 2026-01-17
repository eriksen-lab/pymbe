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
        # number of occupied orbitals
        nocc = 2

        # number of active space orbitals
        ncas = 8

        # number of orbitals
        norb = mol.nao

        # number of electrons in correlated space
        nelecas = (mol.nelec[0] - nocc, mol.nelec[1] - nocc)

        # occupied orbital indices
        occ_idx = slice(nocc)

        # active orbital indices
        cas_idx = slice(nocc, nocc + ncas)

        # occupied + active orbital indices
        occ_cas_idx = slice(nocc + ncas)

        # hf calculation
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # orbsym
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        orbsym = orbsym[cas_idx]

        # expansion space
        exp_space = np.arange(ncas, dtype=np.int64)

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", hf.mo_coeff, hcore_ao, hf.mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph")
        eri = ao2mo.incore.full(eri_ao, hf.mo_coeff, compact=False)

        # inactive Fock matrix elements
        inact_fock = (
            hcore[:, occ_cas_idx]
            + 2 * np.einsum("pqjj->pq", eri[:, occ_cas_idx, occ_idx, occ_idx])
            - np.einsum("pjjq->pq", eri[:, occ_idx, occ_idx, occ_cas_idx])
        )

        # integrals outside correlated space
        eri_goaa = eri[:, occ_idx, cas_idx, cas_idx]
        eri_gaao = eri[:, cas_idx, cas_idx, occ_idx]
        eri_gaaa = eri[:, cas_idx, cas_idx, cas_idx]

        # integrals inside correlated space
        hcore = (
            hcore[cas_idx, cas_idx]
            + 2 * np.einsum("uvii->uv", eri[cas_idx, cas_idx, occ_idx, occ_idx])
            - np.einsum("uiiv->uv", eri[cas_idx, occ_idx, occ_idx, cas_idx])
        )
        eri = eri[cas_idx, cas_idx, cas_idx, cas_idx]

        # create mbe object
        mbe = MBE(
            target="genfock",
            mol=mol,
            norb=ncas,
            nelec=nelecas,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            exp_space=exp_space,
            rst=rst,
            full_norb=norb,
            full_nocc=nocc,
            inact_fock=inact_fock,
            eri_goaa=eri_goaa,
            eri_gaao=eri_gaao,
            eri_gaaa=eri_gaaa,
            no_singles=False,
        )

        # perform calculation
        gen_fock = mbe.kernel()

    else:
        # create mbe object
        mbe = MBE()

        # perform calculation
        gen_fock = mbe.kernel()

    return gen_fock


if __name__ == "__main__":
    # call example function
    gen_fock = mbe_example()

    # finalize mpi
    MPI.Finalize()
