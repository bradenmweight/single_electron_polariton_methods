#!/home/mtayl29/anaconda3/bin/python
import numpy as np
import scipy as sc
# from scipy.special import erf
# from sympy.functions.special.gamma_functions import uppergamma
# from matplotlib import pyplot as plt
# from numpy import kron
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp
from functools import partial
import get_potential_data as v
import solve_hamiltonian as solve

class param:
    wc = 1
    g_wc = 0
    nf = 5
    NCPUS = 48
    nR = 256 # 512
    kGrid = np.zeros(nR)
    omega = 0
    xi_g = 0
    m_0 = 1
    hbar = 1
    ng = 48*2

def test_expm():
    # print("Start")

    nf = 100
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    b = b.T

    n = b.T @ b

    u = sc.linalg.expm(1j* b.T @ b * np.pi / 2.0)

    b_transform = np.conjugate(u.T) @ b.T @ u

    np.savetxt( f"test.dat", b_transform )

    test = np.allclose(b_transform, -1j*b.T, 0,1e-10)

def main():

    constants = param()
    sp.call(f"mkdir -p data", shell=True)

    # test_expm()

    Vk, constants.wc, constants.kGrid =  v.get_V(constants.nR)

    print(f"wc = {constants.wc}")

    g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), constants.ng ))
    with mp.Pool(processes=constants.NCPUS) as pool:
        pool.map(partial(solve.solve_H, Vk, constants), g_wc_list)



if ( __name__ == '__main__' ):
    main()