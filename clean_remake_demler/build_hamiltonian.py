#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from numpy import kron
from numba import jit


def get_shifted_v(i_m,i_ph,Vk,constants):
    # Define constants
    nf = constants.nf
    nR = constants.nR
    xi_g = constants.xi_g
    kGrid = constants.kGrid
    
    # Generate chi
    # b = get_b(nf)
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    b = b.T
    chi = b + b.T

    # Generate the V with phase factor
    v_shifted = np.zeros((nf*nR,nf*nR), dtype=complex)
    
    for k1_ind in range(len(kGrid)):
        for k2_ind in range(len(kGrid)):
            # Define k values
            k1 = kGrid[k1_ind]
            k2 = kGrid[k2_ind]
            kdiff = k1 - k2

            # Define phase term
            phase = chi * xi_g * kdiff * 1j
            exponential = sc.linalg.expm(phase)

            # Define V(kdiff) operator
            v_diff = np.zeros((nR,nR), dtype=complex)
            v_diff[k1_ind,k2_ind]= Vk[ k1_ind-k2_ind ]

            # Add kron(v_diff, exponential) to v_shifted
            v_shifted += kron(v_diff, exponential) / 2.0 / np.pi
            
    return v_shifted

def get_b(nf):
    b = np.zeros((nf,nf), dtype=complex)
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    return b.T

def get_momentum(constants):
    kGrid = constants.kGrid
    m_eff = constants.m_0 * (1.0 + 2.0 * constants.g_wc**2)

    k_e = np.diag(constants.hbar * kGrid**2 / 2.0 / m_eff)

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

def construct_h_total(Vk, constants):
    # Define identities
    nR = constants.nR
    nf = constants.nf
    i_m = np.identity(nR)
    i_ph = np.identity(nf)

    constants.omega, constants.xi_g = get_couplings(constants)

    k_e = get_momentum(constants)
    v_shifted = get_shifted_v(i_m,i_ph,Vk, constants)

    H = np.zeros((nf*nR,nf*nR), dtype=complex)
    H += kron(i_m, get_h_ph(constants))
    H += kron(k_e , i_ph)
    H += v_shifted

    return H