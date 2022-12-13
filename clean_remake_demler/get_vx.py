#!/home/mtayl29/anaconda3/bin/python
import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from scipy.special import erf
#from numba import jit


def get_V_x__201Erfs( nR , r_max):
    RGrid = np.linspace(-r_max,r_max,nR)
    a0 = 20 # Lattice Spacing for Erf(x)/x, Erf( x-a0 ) / (x-a0), a0 = Length (a.u.)
    r0 = 5  # Erf width --> Erf( x/r0 ) / x, r0 = Length (a.u.)
    Vx = np.zeros(( nR ))
    for n in np.arange( -100,101 ):
        Vx += -erf( (RGrid-n*a0) / r0 ) / (RGrid - n*a0)
    return Vx

def get_double_well(nR):
    RGrid = np.linspace(-2.6 / 512 * nR, 2.6 / 512 * nR, nR)
    # RGrid = np.linspace(-1.6, 1.6, nR)
    # RGrid = np.linspace(-2, 2, nR)
    Vx =  np.zeros(len(RGrid))
    beta = 50
    gamma = 95 
    # beta = 3
    # gamma = 3.85 
    Vx = - beta * RGrid**2 / 2 + gamma * RGrid**4 /4
    return RGrid, Vx

def get_QHO(nR):
    RGrid = np.linspace(-5 / 128 * nR, 5 / 128 * nR, nR)
    Vx = np.zeros(( nR ))
    w = 1.0
    Vx += 0.5 * w**2 * RGrid**2
    return RGrid, Vx

def get_square(nR):
    RGrid = np.linspace(-2, 2, nR)
    Vx =  np.zeros(len(RGrid))
    width = 0.5
    
    return RGrid, Vx