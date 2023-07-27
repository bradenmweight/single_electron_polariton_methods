#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from functools import partial
import solve_hamiltonian as solve

# Calculate the dispersion plots for erf(x)

class param:
    wc_norm= 1.0
    wc = 0.0 # This number means nothing
    # g_wc = [0.1, 0.2, 0.3, 1, 10, 100]
    g_wc = [100.]
    nf = 7
    NCPUS = 48
    nk = 1024
    n_kappa = 101 # must be odd 
    n_kappa2 = 11 # must be odd 
    k = 0.0
    a_0 = 4
    Z = 0.1278
    r_0 = 10
    k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)
    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa-1) / 2, (n_kappa-1) / 2, n_kappa)
    kappa_grid2 = 2 * np.pi / a_0 * np.linspace(-(n_kappa2-1) / 2, (n_kappa2-1) / 2, n_kappa2)
    omega = 0
    xi_g = 0
    m_0 = 1
    hbar = 1

def solve_wrapper(constants, g_wc, k):
    return solve.solve_H(constants, k, g_wc)

def main():

    constants = param()
    sp.call(f"mkdir -p data", shell=True)

    print(f"wc = {constants.wc}")

    for gc in constants.g_wc:
        with mp.Pool(processes=constants.NCPUS) as pool:
            pool.map(partial(solve_wrapper, constants, gc), constants.k_points)


if ( __name__ == '__main__' ):
    main()