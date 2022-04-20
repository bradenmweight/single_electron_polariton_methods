#!/home/mtayl29/anaconda3/bin/python
import numpy as np
# import scipy as sc
# from scipy.special import erf
# from sympy.functions.special.gamma_functions import uppergamma
# from matplotlib import pyplot as plt
# from numpy import kron
# from numba import jit
# import subprocess as sp
# import sys
# import multiprocessing as mp
from asyncio import constants
import get_potential_data as v
import solve_hamiltonian as solve

class param:
    wc = 0
    g_wc = 0
    nf = 5
    NCPUS =12
    nR = 128
    kGrid = np.zeros(nR)
    omega = 0
    xi_g = 0


def main():

    constants = param()

    Vmat_k, constants.wc, constants.kGrid =  v.get_V(constants.nR)

    E, U = solve.solve_H(Vmat_k, constants)

    # g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(1000), 48 ) )
    # with mp.Pool(processes=NCPUS) as pool:
    #     pool.map( wrapper_function(), g_wc_list )



if ( __name__ == '__main__' ):
    main()