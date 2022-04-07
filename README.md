# single_electron_polariton_methods

The purpose of this code is to compute exciton-polariton energies and wavefunctions using a variety of Hamiltonians and in a variety of basis sets.

The first step in the run the "get_Potential_Data.py" script. Here, there are a few simple one-electron potentials to choose from, such as the quantum harmonic oscillator (QHO) and a couple double-well potentials from  Ashida et al. PRL 126, 153603 (2021). One can define their own single-electron potential with great ease.

The second step is to run the "One-Electron_Polariton_Solver.py" script, which solves the electron-polariton Hamiltonian in a variety of ways.
  1. Asymptotically Decoupled Hamiltonian -- Ashida et al. PRL 126, 153603 (2021)
      --- Requested by choosing BASIS = Pc (Solves photonic part in its momentum basis)
      --- Requested by choosing BASUS = Fock (Solves photonic part in its Fock state basis defined from b^{\dag} b operators)
 
 The important parameters are the single-mode cavity energy wc (located at the top of the script) and the electron-photon coupling strength A0 which is passed in as an argument (see "example" folder).
 
 
 
 
 
**************************************
Future plans are comprised of the following:
  1. Implement the Pauli-Fierz ("D dot E") Hamiltonian using the Fock basis (or Polarized Fock Basis)
  2. Add options for BASIS of matter and BASIS of photon as well as the Hamiltonanian that is to be solved





################################################
Authors: Braden M. Weight and Michael AD Taylor

Questions/Bugs: bweight@ur.rochester.edu
