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
    ng = 121
    g_wc_grid = 10 ** ( np.linspace( np.log10(10**-2), np.log10(100), ng ))
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

def main(g_min_log, g_max_log, ng, nk, n_kappa):

    constants = param()
    constants.ng = ng
    constants.nk = nk
    constants.n_kappa = n_kappa
    constants.k_points = np.linspace(-np.pi / constants.a_0, np.pi / constants.a_0, nk)
    constants.g_wc_grid = 10 ** ( np.linspace((g_min_log), (g_max_log), ng , endpoint=False))
    sp.call(f"mkdir -p data", shell=True)

    print(f"wc = {constants.wc}")
    print(f"ng = {constants.ng}, gmin = {np.min(constants.g_wc_grid)}, and gmax = {np.max(constants.g_wc_grid)}")

    if len(constants.g_wc_grid) > len(constants.k_points):
        for k in constants.k_points:
            with mp.Pool(processes=constants.NCPUS) as pool:
                pool.map(partial(solve.solve_H, constants, k), constants.g_wc_grid)
    else:
        for gc in constants.g_wc:
            with mp.Pool(processes=constants.NCPUS) as pool:
                pool.map(partial(solve_wrapper, constants, gc), constants.k_points)




if ( __name__ == '__main__' ):
    main(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))