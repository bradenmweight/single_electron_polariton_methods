#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
from scipy.special import erf
from sympy.functions.special.gamma_functions import uppergamma
from matplotlib import pyplot as plt
from numpy import kron
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from main import param
from build_hamiltonian import construct_h_total


def solve_H(const, k, g_wc):
    const.k = k
    const.g_wc = g_wc

    print(f"k-point = {k}")
    H = construct_h_total(const)
    E, U = np.linalg.eigh( H )
    
    wc = const.wc
    # g_wc = const.g_wc
    print(f"g/w_c = {g_wc}")
    np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", E )
    # np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat", E - E[0] )
    # np.savetxt( f"data/E_RAD_k{np.round(k,3)}_{const.nf}_{const.n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )

    return 