#!/home/mtayl29/anaconda3/bin/python
import numpy as np
# import scipy as sc
from numpy import kron
from numba import jit
import scipy as sc
import scipy.special as scsp
import math
from main_disp import param

def get_v(constants):
    # Define constants
    nf = constants.nf
    kappa_grid = constants.kappa_grid
    kappa_grid2 = constants.kappa_grid2
    n_kappa = constants.n_kappa
    n_kappa2 = constants.n_kappa2
    r_0 = constants.r_0
    z = constants.Z
    
    # Generate the V with phase factor
    v = np.zeros((n_kappa,n_kappa), dtype=complex)
    
    # print(f"kappa grid = {kappa_grid2}")

    for kappa_ind2 in range(len(kappa_grid2)):
        for kappa_ind in range(len(kappa_grid)):
            kappa_max = np.max(kappa_grid)
            kappa = kappa_grid[kappa_ind]
            kappa2 = kappa_grid2[kappa_ind2]

            i_ph = np.zeros((nf,nf))
            # print("indexing")

            # Define V(kdiff) operator
            v_diff = np.zeros((n_kappa,n_kappa), dtype=complex)
            if (kappa + kappa2) <= kappa_max and (kappa + kappa2) >= -1 * kappa_max:
                if kappa2 == 0.0:
                    v_diff[kappa_ind, kappa_ind] = 0
                else:
                    shift_ind = kappa_ind2 - ((n_kappa2 - 1) // 2)
                    v_diff[kappa_ind, kappa_ind + shift_ind] = -z / 2 / np.pi * scsp.gammaincc(1e-7, (kappa2 / 2 / r_0)**2) * math.gamma(1e-7)
                    # print("Shifting")

            v += (v_diff)

    return v

def get_a(nf):
    a = np.zeros((nf,nf), dtype=complex)
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T


def get_p_a(constants):
    n_kappa = constants.n_kappa
    nf = constants.nf
    i_m = np.identity(n_kappa, dtype=complex)
    i_ph = np.identity(nf, dtype=complex)

    kappa_grid = constants.kappa_grid
    m = constants.m_0

    a = get_a(nf)

    vector_pot = constants.a_k * (a + a.T)
    print(constants.a_k)

    p_new = constants.hbar * ( kron(np.diag(kappa_grid + constants.k),i_ph) )
    p_new -= kron(i_m , a.T@a * (constants.k - constants.k_shift))
    p_new -= kron(i_m, vector_pot)

    # k_e = np.diag(constants.hbar * (kappa_grid + constants.k)**2 / 2.0 / m)
    p_a = p_new@p_new / 2.0 / m

    return p_a    

def get_h_ph(constants):
    nf = constants.nf
    wc = constants.wc
    h_ph = np.identity(nf)
    for n in range(nf):
        h_ph[n,n] = wc * n
    return h_ph

def get_couplings(constants):
    m = constants.m_0 # Mass of electron
    hbar = constants.hbar
    g = constants.wc * constants.g_wc 

    a_k = g * np.sqrt( hbar * m / constants.wc)
    print(f"a_k = {a_k}")

    return a_k

def construct_h_total(constants):
    # Define identities
    # print("main")
    n_kappa = constants.n_kappa
    nf = constants.nf
    i_m = np.identity(n_kappa)
    i_ph = np.identity(nf)

    constants.a_k = get_couplings(constants)

    p_a = get_p_a(constants)
    v = get_v(constants)

    H = np.zeros((nf*n_kappa,nf*n_kappa), dtype=complex)
    H += kron(i_m, get_h_ph(constants))
    H += p_a
    H += kron(v,i_ph)

    return H
