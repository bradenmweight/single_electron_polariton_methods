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

def construct_h_total(Vk, constants):
    # Define identities
    i_m = np.identity(constants.Rn)
    i_ph = np.identity(constants.nf)

    constants.omega, constants.xi_g = get_couplings(constants)

    k_e = get_momentum(constants)
    v_shifted = get_shifted_v(i_m,i_ph,Vk, constants)

    H = kron(i_m, get_h_ph(constants))
    H += kron(k_e , i_ph)

    return H

def get_shifted_v(i_m,i_ph,Vk,constants):
    # Define constants
    nf = constants.nf
    Rn = constants.Rn
    xi = constants.xi
    
    # Generate chi
    # b = get_b(nf)
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    b = b.T
    chi = b + b.T

    # Generate the V with phase factor
    v_shifted = np.zeros((nf*Rn,nf*Rn))



    chi_full_space = kron(i_m, chi)

    return v_shifted

def get_b(nf):
    b = np.zeros((nf,nf))
    for m in range(1,nf):
        b[m,m-1] = np.sqrt(m)
    return b.T

def get_momentum(constants):
    kGrid = constants.kGrid
    m_eff = constants.m_0 * (1.0 + 2.0 * constants.g_wc**2)

    k_e = np.diag(constants.hbar * kGrid**2 / 2.0 / m_eff)

    return k_e
    

def get_h_ph(constants):
    nf = constants.nf
    omega = constants.omega
    h_ph = np.identity(nf)
    for n in range(nf):
        h_ph[n] = omega * n
    return h_ph

def get_couplings(constants):
    m = constants.m_0 # Mass of electron
    hbar = constants.hbar
    g = constants.wc * constants.g_wc 
    N = 1 # Number of electrons

    omega = np.sqrt(constants.wc**2 + 2 * N * g**2)
    x_omega = np.sqrt(hbar / (m * omega))
    xi_g = g * x_omega / omega

    return omega, xi_g



def solve_H(Vk, constants):
    
    H = construct_h_total(Vk, constants)


    E = 0
    U = 0
    return E, U