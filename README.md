# single_electron_polariton_methods

The purpose of this code is to compute exciton-polariton energies and wavefunctions using a variety of Hamiltonians and in a variety of basis sets.

The first step in the run the "get_Potential_Data.py" script. Here, there are a few simple one-electron potentials to choose from, such as the quantum harmonic oscillator (QHO) and a couple double-well potentials from  Ashida et al. PRL 126, 153603 (2021). One can define their own single-electron potential with great ease.

The second step is to run the "One-Electron_Polariton_Solver.py" script, which solves the electron-polariton Hamiltonian in a variety of ways.
  1. Asymptotically Decoupled Hamiltonian -- Ashida et al. PRL 126, 153603 (2021)
      
      --- Requested by choosing HAM = AD
      
      --- Either BASIS_PHOTON = Pc (Solves photonic part in its momentum basis) or BASIS_PHOTON = Fock (Solves photonic part in its Fock state basis defined from b^{\dag} b operators)
      
      --- Either BASIS_ELECTRON = R or BASIS_ELECTRON = K will solve this Hamiltonian using the respective electronic basis
 
  2. Pauli-Fierz Hamiltonian --- Mandal et al., J. Phys. Chem. Lett. 2020, 11, 9215−9223 [NEED ORIGINAL CITATION FOR PF.]
 
      --- Requested by choosing HAM = PF
      
      --- Requested by choosing BASIS_PHOTON = Fock (Solves photonic part in its Fock state basis defined from a^{\dag} a operators)
      
      --- Requested by BASIS_ELECTRON = R
      
  3. Jaynes-Cummings Hamiltonian
 
      --- Requested by choosing HAM = JC
      
      --- Requested by choosing BASIS_PHOTON = Fock (Solves photonic part in its Fock state basis defined from a^{\dag} a operators)
      
      --- Requested by BASIS_ELECTRON = R
 
 
 The important parameters are the single-mode cavity energy wc (located at the top of the script) and the electron-photon coupling strength A0 which is passed in as an argument (see "SBATCH" submission script).
 
 Some simple plotting routines are provided in the "plot_polaritons.py" script.
 
 
 
**************************************
Future plans are comprised of the following:
  1. Need to add options to do Polarized Fock States (PFS)





################################################

Authors: Braden M. Weight and Michael AD Taylor

Questions/Bugs: bweight@ur.rochester.edu
