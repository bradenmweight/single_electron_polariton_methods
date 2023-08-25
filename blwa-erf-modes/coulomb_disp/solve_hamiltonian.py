#!/home/mtayl29/anaconda3/bin/python
import numpy as np
from scipy.linalg import eigh
# import scipy as sc
# from scipy.special import erf
# from sympy.functions.special.gamma_functions import uppergamma
# from matplotlib import pyplot as plt
from numpy import kron
# from numba import jit
# import subprocess as sp
# import sys
# import multiprocessing as mp
from main_disp import param
from build_hamiltonian import construct_h_total, get_a
import os


def get_filename(k, nf, n_kappa, g_wc, wc_norm):
    return f"data/E_pA_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc_norm,4)}.dat"

def solve_H(const, k, g_wc):
    const.k = k
    const.g_wc = g_wc
    const.wc = np.sqrt( const.wc_norm**2 + (k - const.k_shift)**2)

    filename = get_filename(k,const.nf,const.n_kappa,g_wc,const.wc_norm)

    if not (const.load_existing and os.path.isfile(filename)):
        print(f"k-point = {k}")
        H = construct_h_total(const)
        E, U = eigh( H , driver = "evd")
        
        photon_num_pA = ave_photon_pA(const, U)
        
        # g_wc = const.g_wc
        print(f"g/w_c = {g_wc}")
        np.savetxt( filename, np.column_stack((E, photon_num_pA)),  delimiter=',' )

    return 

def ave_photon_pA(const, U):
    n_kappa = const.n_kappa
    nf = const.nf
    i_m = np.identity(n_kappa)

    a = get_a(nf)

    N = kron(i_m, a.T@a)

    return np.absolute(np.diag( np.conjugate(U.T) @ N @ U))
