#!/home/mtayl29/anaconda3/bin/python
from asyncio import constants
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

def construct_h_total(Vmat_k, constants):
    # Copy relevent constants
    g_wc = constants.g_wc
    wc = constants.wc
    kGrid = constants.kGrid
    Rn = constants.Rn
    nf = constants.nf

    # Define identities
    i_m = np.identity(Rn)
    i_ph = np.identity(nf)

    constants.omega, constants.xi_g = get_couplings(g_wc,wc)

    H = kron(i_m, get_h_ph(constants))
    H += Vmat_k

    return H


def get_h_ph(constants):
    nf = constants.nf
    omega = constants.omega
    h_ph = np.identity(nf)
    for n in range(nf):
        h_ph[n] = omega * n
    return h_ph

def get_couplings(g_wc,wc):
    q = 1.0 # Charge of electron
    m = 1.0 # Mass of electron
    hbar = 1.0
    g = wc * g_wc 
    N = 1 # Number of electrons

    omega = np.sqrt(wc**2 + 2 * N * g**2)
    x_omega = np.sqrt(hbar / (m * omega))
    xi_g = g * x_omega / omega

    return omega, xi_g



def solve_H(Vmat_k, constants):
    
    H = construct_h_total(Vmat_k, constants)


    E = 0
    U = 0
    return E, U