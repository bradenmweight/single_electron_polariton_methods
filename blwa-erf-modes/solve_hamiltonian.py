#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from scipy.special import erf
from sympy.functions.special.gamma_functions import uppergamma
from matplotlib import pyplot as plt
from numpy import kron
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from main import param
from build_hamiltonian import construct_h_total, get_b, get_a


def solve_H(const, k, g_wc):
    const.k = k
    const.g_wc = g_wc
    const.wc = np.sqrt( const.wc_norm**2 + k**2)

    print(f"k-point = {k}")
    H = construct_h_total(const)
    E, U = np.linalg.eigh( H )
    
    photon_num_dE = ave_photon_dE(const, U)
    photon_num_pA = ave_photon_pA(const, U)
    photon_num_AD = ave_photon_AD(const, U)
    
    # g_wc = const.g_wc
    print(f"g/w_c = {g_wc}")
    np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(const.wc_norm,4)}.dat"
               , np.column_stack((E, photon_num_dE,photon_num_pA,photon_num_AD)),  delimiter=',' )
    # np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat", E - E[0] )
    # np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )

    return 

def ave_photon_dE(const, U):
    n_kappa = const.n_kappa
    nf = const.nf
    xi_g = const.xi_g
    i_m = np.identity(n_kappa)
    i_ph = np.identity(nf)

    b = get_b(nf)
    
    kappa_grid = const.kappa_grid
    p = np.diag(const.hbar * (kappa_grid + const.k)**2 / 2.0 / const.m_0)

    N = kron(i_m,b.T @ b) + xi_g * kron(p, b.T + b) + xi_g**2 * kron(p*p, i_ph) / const.hbar

    return np.absolute(np.diag( np.conjugate(U.T) @ N @ U))

def ave_photon_pA(const, U):
    n_kappa = const.n_kappa
    nf = const.nf
    i_m = np.identity(n_kappa)

    a = get_a(nf, const)

    N = kron(i_m, a.T@a)

    return np.absolute(np.diag( np.conjugate(U.T) @ N @ U))

def ave_photon_AD(const, U):
    n_kappa = const.n_kappa
    nf = const.nf
    i_m = np.identity(n_kappa)

    b = get_b(nf)

    N = kron(i_m, b.T@b)

    return np.absolute(np.diag( np.conjugate(U.T) @ N @ U))