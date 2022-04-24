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


def solve_H(Vk, constants, g_wc):
    vcon = constants
    vcon.g_wc = g_wc

    print(f"g/wc = {g_wc}")
    H = construct_h_total(Vk, vcon)
    E, U = np.linalg.eigh( H )
    
    wc = constants.wc
    np.savetxt( f"data/E_AD_{constants.nf}_{constants.nR}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", E )
    np.savetxt( f"data/E_AD_{constants.nf}_{constants.nR}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat", E - E[0] )
    np.savetxt( f"data/E_AD_{constants.nf}_{constants.nR}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )

    return 