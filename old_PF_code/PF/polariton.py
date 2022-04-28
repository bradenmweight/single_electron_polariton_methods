import math
import time
import numpy as np
import scipy as sp
import sys
from numpy import linalg as LA
from numpy import kron as ꕕ
from numpy import array as A
#global xi
#xi = float(sys.argv[1])

def Eig(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#-------------------------------------
#-------------------------------------
def ĉ(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

class param:
    H = A([[0,0],[0,1]])
    µ = A([[0,1],[1,0]])
    ns = 4
    nf = 100
    η_1  = 0.15
    η_3  = 0.15 / np.sqrt(3)
    ωc = -1.5653130645106976 + 1.5848276060994637
    χ_1 = ωc * η_1
    χ_3 = ωc * η_3 * 3

#----------------------------------------
# Data of the diabatic states
def Ĥ(param):
    H = param.H 
    µ = param.µ
    ns = param.ns
    nf = param.nf
    ωc = param.ωc 
    χ_1 = param.χ_1
    χ_3 = param.χ_3
    #------------------------
    Iₚ_1 = np.identity(nf)
    Iₚ_3 = np.identity(nf)
    Iₘ = np.identity(ns)
    #------ Photonic Part -----------------------
    Hₚ_1 = np.identity(nf)
    Hₚ_1[np.diag_indices(nf)] = np.arange(nf) * ωc
    Hₚ_3 = np.identity(nf)
    Hₚ_3[np.diag_indices(nf)] = np.arange(nf) * ωc * 3.0
    â = ĉ(nf) 
    #--------------------------------------------
    #       matter ⊗ photon 
    #--------------------------------------------
    Hij   = ꕕ(H, Iₚ_1)                    # Matter
    Hij  += ꕕ(Iₘ, Hₚ_1)                   # Photon
    Hij  += ꕕ(µ, (â.T + â)) * χ_1         # Interaction
    Hij  += ꕕ(µ @ µ, Iₚ_1) * (χ_1**2/ωc)    # Dipole Self-Energy
    # Hij  += ꕕ(µ @ µ, Iₚ_1) * (χ_3**2/(3.0 * ωc)) # Second mode


    # Hij   = ꕕ(ꕕ(H, Iₚ_1),Iₚ_3)     # Matter
    # Hij  += ꕕ(ꕕ(Iₘ, Hₚ_1),Iₚ_3)     # Photon 1
    # Hij  += ꕕ(ꕕ(Iₘ, Iₚ_1),Hₚ_3)     # Photon 3
    # Hij  += ꕕ(ꕕ(µ, (â.T + â)) * χ_1,Iₚ_3)      # Interaction 1
    # Hij  += ꕕ(ꕕ(µ, Iₚ_1), (â.T + â)) * χ_3     # Interaction 3
    # Hij  += ꕕ(ꕕ(µ @ µ, Iₚ_1),Iₚ_3) * (χ_1**2/ωc)    # Dipole Self-Energy
    # Hij  += ꕕ(ꕕ(µ @ µ, Iₚ_1),Iₚ_3) * (χ_3**2/(3.0 * ωc)) # DSE 3

    return Hij 
#--------------------------------------------------------

#----------------------------------------

if __name__ == "__main__":
 print (Ĥ(param))






