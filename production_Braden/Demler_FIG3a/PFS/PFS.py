import numpy as np
from scipy.linalg import eigh as SC_EXACT_EIGH
from scipy.special import eval_hermite as Hm
from scipy.special import assoc_laguerre as Lg
from scipy.integrate import simpson
import scipy
from math import factorial as fact
from numpy import kron as ꕕ
import sys
from matplotlib import pyplot as plt
import multiprocessing as mp
from numba import jit

def ĉ(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

def getGlobals():
    global Nad, nf, η_1, ωc, ωceV, χ_1, fock_overlap_method, Npol
    global NCPUS

    NCPUS = 5

    Nad = 1 # Number of Electronic Basis States
    nf = 20 # Number of Polarized Fock Basis States

    ωc  = 0.98600303 # a.u.
    χ_1 = 0.007 # PFS Paper Arkajit: \chi = 0.007 a.u. --> \eta = \chi / wc



    #### DONT CHANGE BELOW HERE ####

    ωc = ωceV / 27.2114
    η_1 = χ_1 / ωc
    Npol = ((Nad+1)*nf)

    fock_overlap_method =  "analytical" # "numerical" or "analytical" or "check", "check" compares the two but uses analytic result

def getSmn( m,n,qcm,qcn ):
    ns = Nad + 1

    def fock_overlap_numerical( Fa,Fb,R_Grid ):
        #return np.sum( Fa*Fb ) * (R_Grid[1]-R_Grid[0])
        return abs( simpson( Fa*Fb, R_Grid ) )
    
    def fock_overlap_analytic( m,n,qcm,qcn ):

        dZP = np.sqrt( 1/(2*ωc) )
        qc0 = qcn - qcm
        ξ   = qc0/(4.0 * dZP)
        G   = np.exp(-2*ξ**2.0)
        X   = (2.0*ξ)**2
        if ( m <= n ):
            S_mn = G * (-ξ * 2)**(n-m) * np.sqrt(fact(m)/fact(n)) * Lg(X,m,n-m)  # n < m
        if ( m > n ):
            S_mn = G * (ξ * 2)**(m-n) * np.sqrt(fact(n)/fact(m)) * Lg(X,n,m-n)  # n < m
        return abs(S_mn)

    if( fock_overlap_method == "analytical" ):
        S_mn = fock_overlap_analytic(m,n,qcm,qcn)           # Analytic Overlap of Fock a and Fock b
        return S_mn
    
    elif ( fock_overlap_method == "numerical" ):
        dr = 0.001
        R_Grid = np.arange(-50,50+dr,dr)
        Fm = HO(R_Grid-qcm,ωc,m)                            # Fock State m, real space
        Fn = HO(R_Grid-qcn,ωc,n)                            # Fock State n, real space
        S_mn = fock_overlap_numerical(Fm,Fn,R_Grid)         # Numerical Overlap of Fock m and Fock n
        

        if ( abs(n-m) > 0 and qcm - qcn > 0.0 ):
            plt.plot( R_Grid, Fm, "r", label=f"HO_{m}" )
            plt.plot( R_Grid, Fn, "g", label=f"HO_{n}" )
            plt.legend()
            plt.xlim(-10,10)
            plt.title(f"Displacement = {qcm - qcn}\nOverlap = {S_mn}")
            plt.savefig("HO.jpg")
            plt.clf()
            #exit()

        return S_mn
    
    elif( fock_overlap_method == "check" ):
        dr = 0.001
        R_Grid = np.arange(-50,50+dr,dr)
        Fm = HO(R_Grid-qcm,ωc,m)
        Fn = HO(R_Grid-qcn,ωc,n)
        S_mn_an = fock_overlap_analytic(m,n,qcm,qcn)           # Analytic Overlap of Fock a and Fock b
        S_mn_nu = fock_overlap_numerical(Fm,Fn,R_Grid)      # Numerical Overlap of Fock m and Fock n
        print (f"Positions: {qcm} {qcn}")
        print (f"Overlap: (D,m,n) = ({round(qcm-qcn,5)}, {m},{n})\t{round(S_mn_an,5)}\t\t{round(S_mn_nu,5)}\t\t{round(abs(S_mn_an - S_mn_nu),8)}")
        assert( abs(S_mn_an - S_mn_nu) < 10**-10 ), "Analytic result does not agree with numerical."
        return S_mn_an

def HO(x,w,n): # Gets real-space harmonic oscillator function
    cons1 = 1.0/((2.0**n) * fact(n))**(1/2)
    cons2 = ( w / np.pi )**(1/4)
    exp  = np.exp(- (w*(x**2)/2.0)) 
    hermit = Hm(n, np.sqrt(w) * x)
    F_x = cons1 * cons2 * exp * hermit
    return F_x

def qc(MUii):
    const = np.sqrt( 2.0/ωc**3.0 )
    return -χ_1 * MUii * const

def getĤpol(Had, MU):
    """
    Input: 
        Had, matter hamiltonian (diagonal) energies from electronic structure
        MU,   matter transition dipole matrix from electronic structure
    Output: Hpol, Pauli-Fierz Hamiltonian with single-mode cavity in PFS basis
    """

    ns = Nad + 1
    Hpol = np.zeros((Npol, Npol))

    ##### Transform Matter Hamiltonian to MH Basis #####
    MUii, UMU = np.linalg.eigh(MU)
    H_MH = UMU.T @ Had @ UMU
    #H_MH = UMU @ Had @ UMU.T # THIS ONE IS WRONG

    """
    for i in range(Npol):                                       # Polaritonic Label # 1
        a = int(i/nf)                                           # Matter Label #1
        m = i%nf                                                # Fock label #1
        #print (i,a,m,'of',Npol,ns,nf)
        for j in range(i,Npol):                                 # Polaritonic Label # 2
            b = int(j/nf)                                       # Matter Label #2
            n = j%nf                                            # Fock label #2
            #print ('\t\t',b,n,'of',ns,nf)
            qcm = qc(MUii[a])                                    # Shift of Fock m
            qcn = qc(MUii[b])                                    # Shift of Fock n
            S_mn = getSmn( m,n,qcm,qcn )                        # Overlap of Fock m and Fock n
            Hpol[i,j]  = ωc * (m + 0.50000) * (m==n) * (a==b)   # Energy of Fock State
            Hpol[i,j] += H_MH[a,b]          * (m==n) * (a==b)   # Energy of Matter State
            Hpol[i,j] += H_MH[a,b] * S_mn   * ( a!=b )          # Matter (MH) Interaction Weighted by Fock Overlap
            Hpol[j,i]  = Hpol[i,j]                              # Symmetric, Real-valued Hamiltonian
    """

    for a in range( Nad+1 ): # Matter
        for m in range( nf ): # PFS
            for b in range( Nad+1 ): # Matter
                for n in range( nf ): # PFS
                    polIND1 = a * nf + m
                    polIND2 = b * nf + n
                    qcm = qc(MUii[a])                                    # Shift of Fock m
                    qcn = qc(MUii[b])                                    # Shift of Fock n
                    S_mn = getSmn( m,n,qcm,qcn )                         # Overlap of Fock m and Fock n
                    #print( np.shape(Hpol), a,b,m,n,polIND1,polIND2 )
                    Hpol[polIND1,polIND2]  = ωc * (m + 0.50000) * (m==n) * (a==b)   # Energy of Fock State
                    Hpol[polIND1,polIND2] += H_MH[a,b]          * (m==n) * (a==b)   # Energy of Matter State
                    Hpol[polIND1,polIND2] += H_MH[a,b] * S_mn   * ( a!=b )          # Matter (MH) Interaction Weighted by Fock Overlap
                    Hpol[polIND2,polIND1]  = Hpol[polIND1,polIND2]                              # Symmetric, Real-valued Hamiltonian

    return Hpol

def getAdiab(ri):

    #E_POLARIZATION = np.array([ ( d == 'x' ), ( d == 'y' ), ( d == 'z' ) ])
    #print ("\tLight Polarization:", E_POLARIZATION * 1.0 )

    RGrid = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_ENERGY_GROUND_STATE.dat')[:,0]
    NSteps = len( RGrid )

    Had = np.zeros(( NSteps, Nad + 1, Nad + 1 )) # Need to add one for ground state
    Had[:,0,0] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_ENERGY_GROUND_STATE.dat')[:NSteps,1]
    Had[:,1,1] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_ENERGY_EXCITED_STATE.dat')[:NSteps,1]

    MU = np.zeros(( NSteps, Nad + 1, Nad + 1 ))
    MU[:,0,0] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_DIPOLE_GROUND_STATE.dat')[:NSteps,1]
    MU[:,1,1] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_DIPOLE_EXCITED_STATE.dat')[:NSteps,1]
    MU[:,0,1] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_TRANSITION_DIPOLE.dat')[:NSteps,1]
    MU[:,1,0] = np.loadtxt('LiF_ELECTRONIC_STRUCTURE/ADIABATIC_TRANSITION_DIPOLE.dat')[:NSteps,1]

    return Had[ri], MU[ri]

def SolvePlotandSave(Hpol,Had,ri):

    #for ri in range( NSteps ):
    #print( f"Diagonalizing and saving step {ri}" )
    # Diagonalize polaritonic Hamiltonian and save
    E, U = SC_EXACT_EIGH(Hpol[:,:]) # This is exact solution
    np.savetxt( f"data_PF/Epol_chi{χ_1}_wc{ωceV}_Nad{Nad}_Npfs{nf}_ri{ri}.dat", E )
    np.savetxt( f"data_PF/Upol_chi{χ_1}_wc{ωceV}_Nad{Nad}_Npfs{nf}_ri{ri}.dat", U ) # These can be large
    #np.save( f"data_PF/Upol_chi{χ_1}_wc{ωceV}_Nad{Nad}_Npfs{nf}_ri{ri}.dat", U ) # Smaller
    
    # U[1,:] HAS NOT SIMPLE MEANING. Lowest matter state in Mullikin-Hirsch basis with 1 polarized fock state. BE CAREFUL WITH THIS.
    #np.savetxt( f"data_PF/Char_chi{χ_1}_wc{ωceV}_Nad{Nad}_Npfs{nf}_ri{ri}.dat", U[1,:] ** 2 ) # Photonic Character -- Saves Space Compared to Storing all U

    # Write original E_adiab to file for comparison
    np.savetxt( f"data_PF/Had_ri{ri}.dat", Had[:,:] ) # Save as matrix for simplicity. Remember when plotting.

def main_Serial():

    #for d in ['y']: # CHOOSE POLARIZATION DIRECTION OF DIPOLE MOMENT
    for ri in range(500):
        Had, MU = getAdiab( ri ) 
        Hpol = getĤpol( Had, MU )
        SolvePlotandSave( Hpol, Had, ri )

def main_Parallel(ri):
    print(f"Working on step {ri}")

    #for d in ['z']: # CHOOSE POLARIZATION DIRECTION OF DIPOLE MOMENT
        
    #print (f"Step: {folderNAME}")
    Had, MU = getAdiab( ri ) 
    #print ( f"\tShape of Total Hamiltonian: ({ len(Had) * nf } x { len(Had) * nf }) " )
    Hpol = getĤpol( Had, MU )
    SolvePlotandSave( Hpol, Had, ri )


def test_Snm():
    for i in range( 10 ):
        for j in range( 10 ):
            ri = 0.1*i
            rj = 0
            getSmn(i,j,ri,rj)


if __name__ == "__main__":

    getGlobals()

    print ("\tParameters (η_1, wc):", η_1, ωc)
    

    runList = np.arange( 500 )
    with mp.Pool(processes=NCPUS) as pool:
        pool.map(main_Parallel,runList)
    
    #main_Serial()

    #test_Snm()
