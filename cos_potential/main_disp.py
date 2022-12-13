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
    wc = 1
    g_wc = 0
    nf = 5
    NCPUS = 48
    nk = 128
    n_kappa = 101 # must be odd 
    k = 0.0
    a_0 = 4
    v_0 = 5 / 27.2112
    k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)
    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa-1) / 2, (n_kappa-1) / 2, n_kappa)
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

    with mp.Pool(processes=constants.NCPUS) as pool:
        pool.map(partial(solve_wrapper, constants, constants.g_wc), constants.k_points)


if ( __name__ == '__main__' ):
    main()