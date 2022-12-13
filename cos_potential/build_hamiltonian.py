#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from numpy import kron
from numba import jit


def get_shifted_v(constants):
    # Define constants
    nf = constants.nf
    xi_g = constants.xi_g
    kappa_grid = constants.kappa_grid
    n_kappa = constants.n_kappa
    
    # Generate chi
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    b = b.T
    chi = b + b.T

    # Generate the V with phase factor
    v_shifted = np.zeros((nf*n_kappa,nf*n_kappa), dtype=complex)
    
    k_0 = 2*np.pi / constants.a_0
    print(f"k_0 = {k_0}")
    for kappa_ind in range(len(kappa_grid)):
            # Define phase term
            phase = chi * xi_g * k_0 * 1j
            exponential_1 = sc.linalg.expm(phase)
            exponential_2 = sc.linalg.expm(-1 * phase)

            # Define V(kdiff) operator
            v_diff_1 = np.zeros((n_kappa,n_kappa), dtype=complex)
            if kappa_ind < n_kappa - 1 :
                v_diff_1[kappa_ind,kappa_ind + 1]= constants.v_0 / 2

            v_diff_2 = np.zeros((n_kappa,n_kappa), dtype=complex)
            if kappa_ind > 0:
                v_diff_2[kappa_ind,kappa_ind - 1]= constants.v_0 / 2

            # Add kron(v_diff, exponential) to v_shifted
            v_shifted += kron(v_diff_1, exponential_1) 
            v_shifted += kron(v_diff_2, exponential_2) 
    return v_shifted

def get_b(nf):
    b = np.zeros((nf,nf), dtype=complex)
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    return b.T

def get_momentum(constants):
    kappa_grid = constants.kappa_grid
    m_eff = constants.m_0 * (1.0 + 2.0 * constants.g_wc**2)

    k_e = np.diag(constants.hbar * (kappa_grid + constants.k)**2 / 2.0 / m_eff)

    return k_e
    

def get_h_ph(constants):
    nf = constants.nf
    omega = constants.omega
    h_ph = np.identity(nf)
    for n in range(nf):
        h_ph[n,n] = omega * n
    return h_ph

def get_couplings(constants):
    m = constants.m_0 # Mass of electron
    hbar = constants.hbar
    g = constants.wc * constants.g_wc 
    N = 1 # Number of electrons

    omega = np.sqrt(constants.wc**2 + 2 * N * g**2)
    x_omega = np.sqrt(hbar / (m * omega))
    xi_g = g * x_omega / omega

    return omega, xi_g

def construct_h_total(constants):
    # Define identities
    n_kappa = constants.n_kappa
    nf = constants.nf
    i_m = np.identity(n_kappa)
    i_ph = np.identity(nf)

    constants.omega, constants.xi_g = get_couplings(constants)

    k_e = get_momentum(constants)
    v_shifted = get_shifted_v(constants)

    H = np.zeros((nf*n_kappa,nf*n_kappa), dtype=complex)
    H += kron(i_m, get_h_ph(constants))
    H += kron(k_e , i_ph)
    H += v_shifted

    return H