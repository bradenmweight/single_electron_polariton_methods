#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from functools import partial
import solve_hamiltonian as solve
from time import time

# Calculate the dispersion plots for erf(x)

class param:
    wc_norm= 1
    wc = 0.0 # This number means nothing
    g_wc = [0.0, 0.1, 0.2]
    # g_wc = [0.001]
    a_k = 0 # placeholder
    nf = 5
    NCPUS = 48
    nk = 128
    n_kappa = 101 # must be odd 
    n_kappa2 = 11 # must be odd 
    k = 0.0
    # V parameters
    a_0 = 4
    # a_0 = 10
    Z = 0.1278
    # Z = 0.025
    r_0 = 10
    k_shift = 0 #np.pi/a_0
    # Grids
    k_points = np.linspace(-np.pi / a_0 + k_shift, np.pi / a_0 + k_shift, nk)
    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa-1) / 2, (n_kappa-1) / 2, n_kappa)
    kappa_grid2 = 2 * np.pi / a_0 * np.linspace(-(n_kappa2-1) / 2, (n_kappa2-1) / 2, n_kappa2)
    # Matter parameters
    m_0 = 1
    hbar = 1
    load_existing = False

def solve_wrapper(constants, g_wc, k):
    return solve.solve_H(constants, k, g_wc)

def main():

    constants = param()
    sp.call(f"mkdir -p data", shell=True)

    start = time()

    print(f"wc = {constants.wc_norm}")

    # for gc in constants.g_wc:
    #     with mp.Pool(processes=constants.NCPUS) as pool:
    #         pool.map(partial(solve_wrapper, constants, gc), constants.k_points)

    for gc in constants.g_wc:
        for k in constants.k_points:
            solve_wrapper(constants, gc, k)

    print(f"Run time = {(time() - start) / 60} min")


if ( __name__ == '__main__' ):
    main()