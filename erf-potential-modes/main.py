#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from functools import partial
import solve_hamiltonian as solve

# Calculate the dispersion plots for cos(x)

class param:
    wc_norm= 1
    wc = 1
    g_wc = 0
    ng = 128
    g_wc_grid = np.exp( np.linspace( np.log(10**-2), np.log(100), ng ))
    nf = 5
    NCPUS = 48
    nk = 32
    n_kappa = 101 # must be odd 
    n_kappa2 = 11 # must be odd 
    k = 0.0
    a_0 = 4
    Z = 0.1278
    r_0 = 10
    k_points = np.linspace(0, np.pi / a_0, nk)
    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa-1) / 2, (n_kappa-1) / 2, n_kappa)
    kappa_grid2 = 2 * np.pi / a_0 * np.linspace(-(n_kappa2-1) / 2, (n_kappa2-1) / 2, n_kappa2)
    omega = 0
    xi_g = 0
    m_0 = 1
    hbar = 1

def main():

    constants = param()
    sp.call(f"mkdir -p data", shell=True)

    print(f"wc = {constants.wc}")

    for k in constants.k_points:
        with mp.Pool(processes=constants.NCPUS) as pool:
            pool.map(partial(solve.solve_H, constants, k), constants.g_wc_grid)



if ( __name__ == '__main__' ):
    main()