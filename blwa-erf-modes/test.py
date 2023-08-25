#!/home/mtayl29/anaconda3/bin/python
import numpy as np
# import scipy as sc
from numpy import kron
from numba import jit
import scipy as sc
import scipy.special as scsp
import math
from time import time

class param:
    wc_norm= 1.0
    wc = 1.0 # This number means nothing
    # g_wc = [0.0, 0.1, 0.2, 0.3, 1, 10, 100]
    g_wc = [0.3]
    nf = 7
    NCPUS = 48
    nk = 240
    n_kappa = 101 # must be odd 
    n_kappa2 = 11 # must be odd 
    k = 0.0
    a_0 = 4
    Z = 0.1278
    r_0 = 10
    k_shift = 0 #np.pi / a_0
    k_points = np.linspace(-np.pi / a_0 + k_shift, np.pi / a_0 + k_shift, nk)
    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa-1) / 2, (n_kappa-1) / 2, n_kappa)
    kappa_grid2 = 2 * np.pi / a_0 * np.linspace(-(n_kappa2-1) / 2, (n_kappa2-1) / 2, n_kappa2)
    omega = 0
    xi_g = 0
    m_0 = 1
    hbar = 1
    load_existing = False

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
    
    # print(f"kappa grid = {kappa_grid2}")

    vstart = time()

    for kappa_ind2 in range(len(kappa_grid2)):

        kappa2 = kappa_grid2[kappa_ind2]

        # # Define phase term
        phase = chi * xi_g * kappa2 * 1j
        exponential = sc.linalg.expm(phase)
        shift = -z / 2 / np.pi * scsp.gammaincc(1e-7, (kappa2 / 2 / r_0)**2) * math.gamma(1e-7)

        for kappa_ind in range(len(kappa_grid)):
            kappa_max = np.max(kappa_grid)
            kappa = kappa_grid[kappa_ind]
            
            kappa2 = kappa_grid2[kappa_ind2]

            # Define phase term
            # phase = chi * xi_g * kappa2 * 1j
            # exponential = sc.linalg.expm(phase)

            # Define V(kdiff) operator
            v_diff = np.zeros((n_kappa,n_kappa), dtype=complex)
            if (kappa + kappa2) <= kappa_max and (kappa + kappa2) >= -1 * kappa_max:
                if kappa2 == 0.0:
                    v_diff[kappa_ind, kappa_ind] = 0
                else:
                    shift_ind = kappa_ind2 - ((n_kappa2 - 1) // 2)
                    # v_diff[kappa_ind, kappa_ind + shift_ind] = -z / 2 / np.pi * scsp.gammaincc(1e-7, (kappa2 / 2 / r_0)**2) * math.gamma(1e-7)
                    v_diff[kappa_ind, kappa_ind + shift_ind] = shift
                    # print(f"kappa2 = {kappa2}")
                    # print(f"Off-diagonal term = {v_diff[kappa_ind, kappa_ind + shift_ind]}")
                    # print(f"Shift Index = {shift_ind}")

            # Add kron(v_diff, exponential) to v_shifted
            v_shifted += kron(v_diff, exponential) 
    
    print(f"V time = {time() - vstart}")

    return v_shifted


def get_couplings(constants):
    m = constants.m_0 # Mass of electron
    hbar = constants.hbar
    g = constants.wc * constants.g_wc 

    omega = np.sqrt(constants.wc**2 + 2 * g**2)
    x_omega = np.sqrt(hbar / (m * omega))
    xi_g = g * x_omega / omega

    return omega, xi_g

def exp_in_loop():
    return

def exp_out_loop():
    return

def main():
    constants = param()
    nrounds = 3
    times = 0.0
    for i in range(nrounds):
        start = time()
        get_shifted_v(constants)
        end = time()
        times += end - start
    print(f"Run time = {(times)/nrounds} sec")

if ( __name__ == '__main__' ):
    main()