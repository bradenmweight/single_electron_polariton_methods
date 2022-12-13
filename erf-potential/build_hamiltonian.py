#!/home/mtayl29/anaconda3/bin/python
import numpy as np
# import scipy as sc
from numpy import kron
from numba import jit
import scipy as sc
import scipy.special as scsp
import math

def get_shifted_v(constants):
    # Define constants
    nf = constants.nf
    xi_g = constants.xi_g
    kappa_grid = constants.kappa_grid
    kappa_grid2 = constants.kappa_grid2
    n_kappa = constants.n_kappa
    n_kappa2 = constants.n_kappa2
    r_0 = constants.r_0
    z = constants.Z
    
    # Generate chi
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    b = b.T
    chi = b + b.T

    # Generate the V with phase factor
    v_shifted = np.zeros((nf*n_kappa,nf*n_kappa), dtype=complex)
    
    print(f"kappa grid = {kappa_grid2}")

    for kappa_ind2 in range(len(kappa_grid2)):
        for kappa_ind in range(len(kappa_grid)):
                kappa_max = np.max(kappa_grid)
                kappa = kappa_grid[kappa_ind]
                kappa2 = kappa_grid2[kappa_ind2]

                # Define phase term
                phase = chi * xi_g * kappa2 * 1j
                exponential = sc.linalg.expm(phase)
                

                # Define V(kdiff) operator
                v_diff = np.zeros((n_kappa,n_kappa), dtype=complex)
                if (kappa + kappa2) <= kappa_max and (kappa + kappa2) >= -1 * kappa_max:
                    if kappa2 == 0.0:
                        v_diff[kappa_ind, kappa_ind] = 0
                    else:
                        shift_ind = kappa_ind2 - ((n_kappa2 - 1) // 2)
                        v_diff[kappa_ind, kappa_ind + shift_ind] = -z / 2 / np.pi * scsp.gammaincc(1e-7, (kappa2 / 2 / r_0)**2) * math.gamma(1e-7)
                        print(f"kappa2 = {kappa2}")
                        print(f"Off-diagonal term = {v_diff[kappa_ind, kappa_ind + shift_ind]}")
                        print(f"Shift Index = {shift_ind}")

                # Add kron(v_diff, exponential) to v_shifted
                v_shifted += kron(v_diff, exponential) 
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