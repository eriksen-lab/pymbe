![](doc/logo/pymbe_logo.png "PyMBE")

[![DOI](https://zenodo.org/badge/852764849.svg)](https://zenodo.org/doi/10.5281/zenodo.13752712)

PyMBE: A Many-Body Expanded Correlation Code 
============================================

News
----

* **October 2019**: The PyMBE code is now open source under the MIT license.


Authors
-------

* Dr. Janus Juul Eriksen (Technical University of Denmark, main author)
* Jonas Greiner (Johannes Gutenberg University Mainz, author). 

Prerequisites
-------------

* [Python](https://www.python.org/) 3.7 or higher
* [PySCF](https://pyscf.github.io/) 1.6 or higher and its requirements
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/) 3.0.0 or higher built upon an MPI-3 library
* [pytest](https://docs.pytest.org/) for unit and system tests


Features
--------

* Many-Body Expanded Full Configuration Interaction.
    - Calculations of Ground State Correlation Energies, Excitation Energies, (Transition) Dipole Moments, 1- and 2-Electron Reduced Density Matrices, and Generalized Fock Matrices.
    - Coupled Cluster Intermediate Base Models: CCSD, CCSD(T), CCSDT, CCSDT(Q), and CCSDTQ.
    - Arbitrary and Automatically Selected Reference Spaces.
    - Clustered Expansion Spaces.
    - Error Control through Orbital-based Screening.
    - Full Abelian Point Group Symmetry in Delocalized Orbital Bases and Arbitrary Point Group Symmetry in Localized Orbital Bases.
    - Closed- and Open-Shell Implementations.
    - Restricted Hartree-Fock (RHF/ROHF) and Complete Active Space Self-Consistent Field (CASSCF) Molecular Orbitals.
* Massively Parallel Implementation.
* MPI-3 Shared Memory Parallelism.


Usage
-----

PyMBE can be installed through pip:
```sh
pip install .
```
PyMBE also requires the PYTHON_HASH_SEED environment variable to be set to 0 (export PYTHON_HASH_SEED=0) for reproducibility reasons and will complain otherwise. Furthermore, the mpi4py implementation of the MPI standard needs to be installed, built upon an MPI-3 library.\
Once these requirements are satisfied, PyMBE can be started by importing the MBE class and creating a MBE object while passing input data and keywords as keyword arguments. Possible keyword arguments are:

* **expansion model**
    * method: electronic structure method (fci, ccsdtq, ccsdt(q) ccsdt, ccsd(t), ccsd)
    * cc_backend: coupled-cluster backend (pyscf, ecc, ncc)
- **target property**
    - target: expansion target property (energy, dipole, excitation, trans, rdm12, genfock)
* **system**
    * mol: [pyscf](https://pyscf.github.io/) gto.Mole object
    * norb: number of orbitals
    * nelec: number of electrons
    * point_group: point group
    * orbsym: orbital symmetry
    * fci_state_sym: state wavefunction symmetry
    * fci_state_root: target state
    * fci_state_weights: weights for state-averaged calculations
    * full_norb: number of orbitals of the entire system (important for generalized fock matrix calculations)
    * full_nocc: number of occupied orbitals of the entire system (important for generalized fock matrix calculations)
- **orbital representation**
    - orb_type: orbital representation
* **integrals**
    * hcore: core hamiltonian integrals
    * eri: electron repulsion integrals
    * dipole_ints: dipole integrals
    * inact_fock: inactive fock matrix (important for generalized fock matrix calculations)
    * eri_goaa: general-occupied-active-active integrals (important for generalized fock matrix calculations)
    * eri_gaao: general-active-active-occupied integrals (important for generalized fock matrix calculations)
    * eri_gaaa: general-active-active-active integrals (important for generalized fock matrix calculations)
- **reference space and expansion spaces**
    - ref_space: reference space
    - exp_space: expansion space
    - ref_thres: quantum fidelity threshold for automatic reference space identification
* **base model**
    * base_method: base model electronic structure method (ccsdtq, ccsdt(q), ccsdt, ccsd(t), ccsd)
    * base_prop: base model property
- **screening**
    - screen_type: screening method (fixed and adaptive)
    - screen_start: screening start order for fixed screening
    - screen_perc: screening percentage for fixed screening
    - screen_func: screening function for fixed screening (max, sum, and sum_abs)
    - screen_thres: error threshold for adaptive screening
    - max_order: maximum expansion order
* **restart**
    * rst: restart logical
    * rst_freq: restart frequency
- **verbose**
    - verbose: verbose option (0: only error output, 1: standard output, 2: screening output, 3: MBE debug output, 4: backend debug output)
* **backends**
    * fci_backend: pyscf fci backend (direct_spin0, direct_spin1, direct_spin0_symm, direct_spin1_symm)
    * cc_backend: cc backend (pyscf, ecc, and ncc)
- **settings**
    - hf_guess: hartree-fock initial guess
    - dryrun: boolean to start dryrun without increment calculations
    - no_singles: omit single excitations

Many of these arguments have default values set or are optional depending on the calculation type. See the [examples](pymbe/examples/) section for a range of example scripts.\
The calculation is started by calling the kernel() member function of the MBE object. Restart files are automatically generated (unless otherwise requested through the rst keyword argument) in a dedicated directory `rst` within `$WORKDIR`, which is deleted in case of successful termination of PyMBE. When restarting a calculation from the restart files, the kernel function can be called without passing any keyword arguments to the MBE object. The kernel() function returns the total target property. The program can be called in parallel by calling the kernel function in multiple MPI processes (e.g. using the mpiexec command). Only the keyword arguments of the MBE object on the global master will be used during the calculation.\
PyMBE can be used for CASCI calculations in two possible ways: The number of orbitals, number of electrons, and integrals for the whole system can be passed to PyMBE and the CAS can be selected using the ref_space and exp_space keywords. If all integrals cannot be kept in memory, only the number of orbitals, number of electrons, and integrals of the CAS can be passed to PyMBE. This requires the HF property of the occupied orbitals outside the CAS to be added to the final property. Additionally, the one-electron Hamiltonian has to be modified by adding the HF potential of the occupied orbitals outside the CAS.\
Expansion space orbitals can be clustered by calling the cluster_orbs() member function of the MBE object. This function will automatically perform an MBE up to the screening criteria specified in the MBE object and then construct an orbital pair correlation matrix which is used for clustering. The maximum cluster size can be passed using the max_cluster_size keyword argument. The most correlated orbitals can be left as single-orbital clusters using the n_single_orbs keyword argument. The symm_eqv_sets keyword argument can be used to pass symmetry-equivalent sets among which orbitals should be clustered. After successful completion, the cluster_orbs function will return the clustered expansion space and the generated orbital pair correlation matrix. Additionally, the expansion space of the MBE object is automatically updated.\
Please note that all PyMBE functions only return the electronic part of the calculated property. For energies and dipole moments the nuclear part has to be added to get the total molecular property. The results of a PyMBE calculation can be printed using the results() member function of the MBE object. This function returns the calculation parameters summarized in a string.\
The results can also be plotted using the plot() member function of the MBE object. This function returns a [matplotlib](https://matplotlib.org) figure.Figure object.


Documentation
-------------

The PyMBE code is deliberately left undocumented, but the [pytest](https://docs.pytest.org/) unit tests and the type hints using the Python [typing](https://docs.python.org/3/library/typing.html) module can be used to comprehend the usage of the different functions and their arguments.


Tutorials
---------

None at the moment, but please have a look at the various 
[examples](pymbe/examples/) that accompany the code.


Citing PyMBE
------------

The following papers document the development of the PyMBE code and the theory 
it implements:

* Error Control and Automatic Detection of Reference Active Spaces in Many-Body Expanded Full Configuration Interaction\
Greiner, J., Gauss, J., Eriksen, J. J.\
[J. Phys. Chem. A, 128, 6806 (2024)](https://pubs.acs.org/doi/10.1021/acs.jpca.4c04056) ([arXiv:2406.11343](https://arxiv.org/abs/2406.11343))

* MBE-CASSCF Approach for the Accurate Treatment of Large Active Spaces\
Greiner, J., Gianni, I., Nottoli, T., Lipparini, F., Eriksen, J. J., Gauss, J.\
[J. Chem. Theory Comput., 20, 4663 (2024)](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00388) ([arXiv:2403.17836](https://arxiv.org/abs/2403.17836))

* Incremental Treatments of the Full Configuration Interaction Problem\
Eriksen, J. J., Gauss, J.\
[WIREs Comput. Mol. Sci., e1525 (2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/wcms.1525?af=R) ([arXiv:2012.07371](https://arxiv.org/abs/2012.07371))

* Ground and Excited State First-Order Properties in Many-Body Expanded Full Configuration Interaction\
Eriksen, J. J., Gauss, J.\
[J. Chem. Phys., 153, 154107 (2020)](https://aip.scitation.org/doi/10.1063/5.0024791) ([arXiv:2008.03610](https://arxiv.org/abs/2008.03610))

* Generalized Many-Body Expanded Full Configuration Interaction Theory\
Eriksen, J. J., Gauss, J.\
[J. Phys. Chem. Lett., 10, 7910 (2019)](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b02968) ([arXiv:1910.03527](https://arxiv.org/abs/1910.03527))

* Many-Body Expanded Full Configuration Interaction. II. Strongly Correlated Regime\
Eriksen, J. J., Gauss, J.\
[J. Chem. Theory Comput., 15, 4873 (2019)](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00456) ([arXiv: 1905.02786](https://arxiv.org/abs/1905.02786))

* Many-Body Expanded Full Configuration Interaction. I. Weakly Correlated Regime\
Eriksen, J. J., Gauss, J.\
[J. Chem. Theory Comput., 14, 5180 (2018)](https://pubs.acs.org/doi/10.1021/acs.jctc.8b00680) ([arXiv: 1807.01328](https://arxiv.org/abs/1807.01328))

* Virtual Orbital Many-Body Expansions: A Possible Route Towards the Full Configuration Interaction Limit\
Eriksen, J. J., Lipparini, F.; Gauss, J.\
[J. Phys. Chem. Lett., 8, 4633 (2017)](https://pubs.acs.org/doi/10.1021/acs.jpclett.7b02075) ([arXiv: 1708.02103](https://arxiv.org/abs/1708.02103))


Bug reports
-----------

Janus Juul Eriksen: janus [at] kemi [dot] dtu [dot] dk

