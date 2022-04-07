import numpy as np
import scipy as sc
from scipy.special import erf
from sympy.functions.special.gamma_functions import uppergamma
from matplotlib import pyplot as plt
from numpy import kron
from numba import jit
import subprocess as sp
import sys

def get_Globals():
    global wc_ev, wc, OMEGA, NKn, f, X_m, X_im, m0, g, A0, Nf, BASIS

    A0 = float(sys.argv[1]) #0.0 # Light-Matter Coupling Strength
    #wc_ev = 0.1 # Photon Energy, eV
    wc = 0.986 # a.u. # Demler Fig 3a E1-E0 matter transition

    m0 = 1.0
    qe = 1.0

    BASIS = "Pc" # "Pc" or "Fock" -- Defines basis of photonic states.
    # Add secondary basis choice for the matter part. Currently, always K-space solver.
    #   Need to be able to solve Pauli-Fierz ("D dot E") as well using dipoles.
    #   Also, we should add the Jaynes-Cummings solution just to satisfy class requirements...

    Nf = 100 # Only used if calculating result in Fock basis




    # DO NOT CHANGE BELOW HERE -- Used only for BASIS = "Pc" option
    #wc = wc_ev / 27.2114
    N = 1 # "Particle Number" -- Not exactly sure what this is...
    g = qe * A0 * np.sqrt( wc / m0 )
    OMEGA = np.sqrt( wc**2 + 2*N*g**2 )
    #print( f"OMEGA = {OMEGA}" )
    X_m = np.sqrt( 1 / OMEGA / m0 )
    X_im = g * X_m / OMEGA


    ### BOOK KEEPING ###
    sp.call("mkdir -p data", shell=True)

def get_b():
    b = np.zeros((Nf,Nf))
    for m in range(1,Nf):
        b[m,m-1] = np.sqrt(m)
    return b.T

def qc2_nm( n, m, pc_Grid ): # DVR Basis for photon position squared
    dpc = pc_Grid[1] - pc_Grid[0]

    norm = (-1) ** (n-m) / dpc**2
    if ( n == m ):
        qc2_nm = np.pi**2 / 3 
    else:
        qc2_nm = 2 / (n-m)**2
    
    return  qc2_nm * norm

def get_H_Total_PcBasis( KGrid, pc_Grid, VMat_k ):

    H_Total = np.zeros(( len(KGrid)*len(pc_Grid), len(KGrid)*len(pc_Grid) ), dtype=complex)
    m_eff = m0 * ( 1 + (g/wc)**2 )

    for K_ind1, K1 in enumerate( KGrid ): # K1
        #print( f"K1 = {K_ind1} of {len(KGrid)}" )
        for Pc_ind1, Pc1 in enumerate( pc_Grid ): # P1
            for K_ind2, K2 in enumerate( KGrid ): # K2
                for Pc_ind2, Pc2 in enumerate( pc_Grid ): # P2

                    index_total_1 = K_ind1 * len(pc_Grid) + Pc_ind1
                    index_total_2 = K_ind2 * len(pc_Grid) + Pc_ind2

                    # Electron Kinetic Energy -- Diagonal in K basis
                    if ( K_ind1 == K_ind2 and Pc_ind1 == Pc_ind2 ):
                        H_Total[index_total_1, index_total_2] += KGrid[K_ind1] ** 2 / 2 / m_eff
                    
                    # Photon Hamiltonian Energy -- DVR for qc ~ b^{dag} + b, diagonal in pc
                    if ( K_ind1 == K_ind2 ):
                        if ( Pc_ind1 == Pc_ind2 ):
                            H_Total[index_total_1, index_total_2] += 0.5 * pc_Grid[ Pc_ind1 ] ** 2
                        H_Total[index_total_1, index_total_2] += 0.5 * OMEGA**2 * qc2_nm( Pc_ind1, Pc_ind2, pc_Grid )

                    # Interaction Term
                        # Diagonal in photon: Only pc appears here, no DVR necessary.
                        # Purely off-diagonal in matter, V(\hat{X}) ~ V(k1 - k2)
                    if ( Pc_ind1 == Pc_ind2 and K_ind1 != K_ind2 ):
                        PI = np.sqrt(2/OMEGA) * Pc_ind1
                        H_Total[ index_total_1, index_total_2 ] += VMat_k[K_ind1,K_ind2] * np.exp( 1j * (K1-K2) * X_im * PI )

    return H_Total

def get_H_Total_FockBasis( KGrid, VMat_k ):

    H_Total = np.zeros(( len(KGrid)*Nf, len(KGrid)*Nf ), dtype=complex)
    m_eff = m0 * ( 1 + (g/wc)**2 )

    op_b = get_b()
    PI_mat = np.sqrt( 2/ OMEGA ) * (op_b.T - op_b)
    matrix_term = sc.linalg.expm( 1j * X_im * PI_mat )

    #### FOR DEBUGGING, PLOT THE INTERACTION MATRICES #### 
    plt.imshow( np.abs(PI_mat), vmin=0.0 )
    plt.colorbar()
    plt.savefig(f"data/PI_mat_Nf{Nf}_wc{wc}_A0{np.round(A0,2)}.jpg")
    plt.clf()
    plt.imshow( np.abs(matrix_term), vmin=0.0 )
    plt.colorbar()
    plt.savefig(f"data/matrix_term_Nf{Nf}_wc{wc}_A0{np.round(A0,2)}.jpg")
    plt.clf()
    ######################################################

    for K_ind1, K1 in enumerate( KGrid ): # K1
        #print( f"K1 = {K_ind1} of {len(KGrid)}" )
        for n in range( Nf ): # P1
            for K_ind2, K2 in enumerate( KGrid ): # K2
                for m in range( Nf ): # P2

                    index_total_1 = K_ind1 * Nf + n
                    index_total_2 = K_ind2 * Nf + m

                    # Electron Kinetic Energy -- Diagonal in K basis
                    if ( K_ind1 == K_ind2 and n == m ):
                        H_Total[index_total_1, index_total_2] += KGrid[K_ind1] ** 2 / 2 / m_eff
                    
                    # Photon Hamiltonian Energy -- Diagonal in Fock Basis -- b^{\dag}b ~ w * n (n==m)
                    if ( K_ind1 == K_ind2 and n == m ):
                        H_Total[index_total_1, index_total_2] += OMEGA * n

                    # Interaction Term:
                        # Off-diagonal in photon: Fock Basis
                        # Off-diagonal in matter, V(\hat{X}) ~ V(k1 - k2)
                    #if ( n != m and K_ind1 != K_ind2 ):
                    #    H_Total[ index_total_1, index_total_2 ] += VMat_k[K_ind1,K_ind2] * matrix_term[n,m]
                    if ( K_ind1 != K_ind2 ):
                        H_Total[ index_total_1, index_total_2 ] += VMat_k[K_ind1,K_ind2] * matrix_term[n,m]

    return H_Total


def get_Reciprocal_space_data():
    VMat_k = np.loadtxt( "Vx/VMat_k.dat", dtype=complex )
    KGrid  = np.loadtxt( "Vx/KGrid.dat" )
    return KGrid, VMat_k

def get_Pc_Grid():
    pc_Grid = np.linspace( -10,10,100 )
    #pc_Grid = np.arange( -10,10,0.1 )
    return pc_Grid

def plot_H( H_Total, name="H_Total.jpg" ):
    EZero_vec = np.ones(( len(H_Total) )) * H_Total[0,0]
    plt.contourf(  np.log( np.abs( H_Total - np.diag( H_Total[np.diag_indices(len(H_Total))] - EZero_vec ) ) + 0.01 ) )
    plt.colorbar()
    plt.title( "Log[|H|]", fontsize=15)
    plt.savefig("data/H_Total.jpg", dpi=500)
    plt.clf()

def main():

    get_Globals()

    # Get matter grid (K) and potential (Vk)
    KGrid, VMat_k = get_Reciprocal_space_data() # Returns basis for momentum space as well as potential in complex momentum space V(k)

    # Get photonic grid: pc
    pc_Grid = get_Pc_Grid()

    if ( BASIS == "Pc" ): #### SOVE IN Pc BASIS ####
        print( f"Size of matrix: {len(KGrid)}*{len(pc_Grid)} = {len(KGrid)*len(pc_Grid)}" )
        H_Total_PcBasis = get_H_Total_PcBasis( KGrid, pc_Grid, VMat_k )
        #plot_H( H_Total_PcBasis, name="H_Total_PcBasis.jpg" )
        E, U = np.linalg.eigh( H_Total_PcBasis )

        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS}.dat", U )


    elif( BASIS == "Fock" ): #### SOVE IN FOCK BASIS ####
        print( f"Size of matrix: {len(KGrid)}*{Nf} = {len(KGrid)*Nf}" )
        H_Total_FockBasis = get_H_Total_FockBasis( KGrid, VMat_k )
        #plot_H( H_Total_FockBasis, name="H_Total_FockBasis.jpg" )
        E, U = np.linalg.eigh( H_Total_FockBasis )

        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"data/E_{BASIS}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS}.dat", U )




if ( __name__ == '__main__' ):
    main()

