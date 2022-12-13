import numpy as np
import scipy as sc
from math import factorial as fact
from scipy.special import erf
from scipy.special import eval_hermite as Hm
from scipy.special import assoc_laguerre as Lg
from scipy.integrate import simpson
from sympy.functions.special.gamma_functions import uppergamma
from matplotlib import pyplot as plt
from numpy import kron
from numba import jit
import subprocess as sp
import sys
import multiprocessing as mp


def get_Globals():
    global wc, m0, qe, Nf, BASIS_ELECTRON, BASIS_PHOTON, HAM, DATA_DIR, Npc
    global NCPUS, NTrunc

    wc = 0.98600303 #/ 27.2114 # a.u. # Demler Fig 3a E1-E0 matter transition
    #wc = 1.03202434509 #/ 27.2114 # a.u. # QHO E1-E0 matter transition

    m0 = 1.0
    qe = 1.0

    HAM = "PFS" # "AD", "PF", "PFS" -- Solves the Asympotically Decoupled (AD) or Pauli-Fierz Hamiltonians
    BASIS_PHOTON   = "Fock" # "Pc" or "Fock" -- Defines basis of photonic states.
    BASIS_ELECTRON = "R" # "R" or "K" -- Defines basis of electronic states as real ("R") or reciprocal ("K").
    
    NCPUS = 36

    ##### TO-DO ##### 
    ###   Need to be able to solve Pauli-Fierz ("D dot E") as well using dipoles. ---> DONE ~ BMW

    Nf  = int(sys.argv[1])  # Only used if calculating result in Fock basis
    Npc = 32 # Only used if calculating result in Grid/DVR basis

    NTrunc = int(sys.argv[2]) # Includes S0 and S1 for NTrunc = 2





    ### BOOK KEEPING ###
    DATA_DIR = f"data_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    #np.savetxt(f"{DATA_DIR}/XI_g_data_{A0}.dat", np.array([ g/wc, X_im ]) )

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

@jit(nopython=True)
def qc2( pc_Grid ): # DVR Basis for photon position squared
    dpc = pc_Grid[1] - pc_Grid[0]
    qc = np.zeros(( len(pc_Grid), len(pc_Grid) ))

    for n in range( len(pc_Grid) ):
        for m in range( len(pc_Grid) ):
            norm = (-1) ** (n-m) / dpc**2
            if ( n == m ):
                qc2[n,m] = np.pi**2 / 3 
            else:
                qc2[n,m] = 2 / (n-m)**2
            qc2[n,m] *= norm
    return  qc2

@jit(nopython=True)
def T_el( RGrid ): # DVR Basis for electronic kinetic energy
    dR = RGrid[1] - RGrid[0]
    T = np.zeros(( len(RGrid), len(RGrid) ))

    for n in range( len(RGrid) ):
        for m in range( len(RGrid) ):
            norm = (-1) ** (n-m) / dR**2
            if ( n == m ):
                T[n,m] = np.pi**2 / 3 
            else:
                T[n,m] = 2 / (n-m)**2
            T[n,m] *= norm
    return  T

def get_H_Total_PcBasis( KGrid, pc_Grid, VMat_k ): # Bad function. Do not use.

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

def get_H_Total_Fock_Reciprocal__PI_SHIFT__FFT_CONVOLUTION( KGrid, VMat_k ):

    H_Total = np.zeros(( len(KGrid)*Nf, len(KGrid)*Nf ), dtype=complex)
    m_eff = m0 * ( 1 + (g/wc)**2 )

    op_b = get_b()

    #### FOR DEBUGGING, PLOT THE INTERACTION MATRICES #### 
    """
    #PI_mat = np.sqrt( 2/ OMEGA ) * (op_b.T - op_b)
    #matrix_term = sc.linalg.expm( 1j * X_im * PI_mat )
    plt.imshow( np.abs(PI_mat), vmin=0.0 )
    plt.colorbar()
    plt.savefig(f"data/PI_mat_Nf{Nf}_wc{wc}_A0{np.round(A0,2)}.jpg")
    plt.clf()
    plt.imshow( np.abs(matrix_term), vmin=0.0 )
    plt.colorbar()
    plt.savefig(f"data/matrix_term_Nf{Nf}_wc{wc}_A0{np.round(A0,2)}.jpg")
    plt.clf()
    """
    ######################################################

    for K_ind1, K1 in enumerate( KGrid ): # K1
        #print( f"K1 = {K_ind1} of {len(KGrid)}" )
        for n in range( Nf ): # Fock 1
            for K_ind2, K2 in enumerate( KGrid ): # K2
                for m in range( Nf ): # Fock 2

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
                        #kdiff = np.abs( K2 - K1 )
                        kdiff = K2 - K1
                        PI_mat = 1j * (op_b.T - op_b)
                        matrix_term = sc.linalg.expm( 1j * kdiff * X_im * PI_mat )
                        H_Total[ index_total_1, index_total_2 ] += (VMat_k[K_ind1,K_ind2] * matrix_term[n,m]) / 2 / np.pi

    return H_Total

def get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION__OLD( KGrid, VMat_k, op_b, OMEGA, Xi, m_eff ):

    H_Total = np.zeros(( len(KGrid)*Nf, len(KGrid)*Nf ), dtype=np.complex128 )

    for K_ind1, K1 in enumerate( KGrid ): # K1
        #print( f"K1 = {K_ind1} of {len(KGrid)}" )
        for n in range( Nf ): # Fock 1
            for K_ind2, K2 in enumerate( KGrid ): # K2
                for m in range( Nf ): # Fock 2

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
                    if ( K_ind1 != K_ind2 ):
                        #kdiff = np.abs( K2 - K1 )
                        kdiff = K2 - K1
                        CHI_mat = op_b.T + op_b
                        matrix_term = sc.linalg.expm( 1j * kdiff * Xi * CHI_mat )
                        H_Total[ index_total_1, index_total_2 ] += VMat_k[K_ind1,K_ind2] * matrix_term[n,m]

    return H_Total

def get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION( KGrid, VMat_k, op_b, OMEGA, Xi, m_eff ):

    def get_v_shifted(KGrid,VMat_k,op_b,Xi):
        
        def braden_kron( A, B, NA, NB ):
            #A = A[:,np.newaxis,:,np.newaxis]
            #A = (A[:,np.newaxis,:,np.newaxis]*B[np.newaxis,:,np.newaxis,:]).reshape( NA*NB,NA*NB )
            A = (A[:,None,:,None]*B[None,:,None,:]).reshape( NA*NB,NA*NB )
            return A

        v_shifted = np.zeros((len(KGrid)*Nf,len(KGrid)*Nf), dtype=np.complex64)
        v_diff = np.zeros(( len(KGrid), len(KGrid) ), dtype=np.complex64)
        chi = op_b + op_b.T
        k1_OLD = 0
        k2_OLD = 0
        for k_ind1, k1 in enumerate(KGrid):
            #print( k_ind1 )
            for k_ind2, k2 in enumerate(KGrid):
                kdiff = k1 - k2
                phase = chi * Xi * kdiff
                exponential = sc.linalg.expm( 1j * phase )
                
                v_diff = np.zeros(( len(KGrid), len(KGrid) ), dtype=np.complex64)
                v_diff[k_ind1,k_ind2] = VMat_k[k_ind1,k_ind2]
                #v_shifted += kron( v_diff, exponential )
                v_shifted += braden_kron( v_diff, exponential, len(KGrid), Nf )

                k1_OLD = k_ind1 * 1
                k2_OLD = k_ind2 * 1

        return v_shifted

    def get_H_ph():
        return np.diag( np.arange( 0,Nf )*OMEGA )

    def get_KE(KGrid,m_eff):
        return np.diag( KGrid**2 / 2 / m_eff )

    # Define Identities in Subspaces
    I_m  = np.identity( len(KGrid) )
    I_ph = np.identity( Nf )

    # Get shifted V from FFT tricks
    V_shifted = get_v_shifted(KGrid,VMat_k,op_b,Xi)

    # Build Ham.
    H_Total = np.zeros(( len(KGrid)*Nf, len(KGrid)*Nf ), dtype=np.complex128 )
    H_Total += kron( I_m, get_H_ph() )
    H_Total += kron( get_KE(KGrid,m_eff), I_ph )
    H_Total += V_shifted

    return H_Total

def get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION__Wrapper( g_wc ):

    def compute_photon_number_AD_basis( N_AD, U, op_b, I_m ):
        op_b_Total = kron( I_m, op_b.T @ op_b )
        for j in range( len(N_AD) ):
            N_AD[j] = U[:,j].T @ op_b_Total @ U[:,j]
        return N_AD

    def compute_photon_number_pA_basis( N_AD_in_pA, U, op_b, I_m ):
        op_b_Total = kron( I_m, op_b.T @ op_b )
        
        v2 = 0.25 * ( (OMEGA**2 + wc**2)/(wc * OMEGA) - 2 )
        adaga = 0.5 * (OMEGA**2 + wc**2)/(OMEGA * wc) * op_b.T @ op_b - 0.25 * (OMEGA**2 - wc**2)/(OMEGA * wc) * (op_b) + v2
        adaga_Total = kron( I_m, adaga )
        
        for j in range( len(N_AD_in_pA) ):
            N_AD_in_pA[j] = U[:,j].T @ adaga_Total @ U[:,j]
        return N_AD_in_pA

    print(f"g_wc = {np.round(g_wc,7)}")

    g = wc * g_wc

    N = 1 # "Particle Number" -- Not exactly sure what this is...
    #g = qe * A0 * np.sqrt( wc / m0 )
    OMEGA = np.sqrt( wc**2 + 2*N*g**2 )
    X_omega = np.sqrt( 1 / OMEGA / m0 )
    Xi = g * X_omega / OMEGA
    m_eff = m0 * ( 1 + (g/wc)**2 )

    KGrid, VMat_k = get_Reciprocal_space_data() # Returns basis for momentum space as well as potential in complex momentum space V(k)

    #print( f"Size of matrix: {len(KGrid)}*{Nf} = {len(KGrid)*Nf}" )
    H_Total = get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION( KGrid, VMat_k, get_b(), OMEGA, Xi, m_eff )
    #print(f"Diagonalizing Hamiltonian.\n")
    E, U = np.linalg.eigh( H_Total )
    print(f"\tFinished g_wc = {g_wc}")

    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", E )
    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat", E - E[0] )
    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
    #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    ### COMPUTE VARIOUS OBSERVABLES ###
    N_AD = np.zeros(( len(E) ))
    N_AD = compute_photon_number_AD_basis( N_AD, U, get_b(), np.identity( len(E)//Nf ) )
    np.savetxt( f"{DATA_DIR}/N_ADBasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", N_AD )
    
    N_AD_in_pA = np.zeros(( len(E) ))
    N_AD_in_pA = compute_photon_number_pA_basis( N_AD_in_pA, U, get_b(), np.identity( len(E)//Nf ) )
    np.savetxt( f"{DATA_DIR}/N_AD_in_pABasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", N_AD_in_pA )
    

def get_H_Total_Fock_Real__CHI_SHIFT( RGrid, Vx_1D, op_b, T_elec ):

    H_Total = np.zeros(( len(RGrid)*Nf, len(RGrid)*Nf ), dtype=complex )
    m_eff = m0 * ( 1 + (g/wc)**2 )
    dR = RGrid[1] - RGrid[0]

    for R_ind1, R1 in enumerate( RGrid ): # R1
        #print( f"K1 = {R_ind1} of {len(RGrid)}" )
        for n in range( Nf ): # Fock 1
            for R_ind2, R2 in enumerate( RGrid ): # R2
                for m in range( Nf ): # Fock 2

                    index_total_1 = R_ind1 * Nf + n
                    index_total_2 = R_ind2 * Nf + m

                    # Electron Kinetic Energy -- DVR Basis using RGrid from file
                    H_Total[index_total_1, index_total_2] += T_elec[R_ind1, R_ind2] / 2 / m_eff
                    
                    # Photon Hamiltonian Energy -- Diagonal in Fock Basis -- b^{\dag}b ~ w * n (n==m)
                    if ( R_ind1 == R_ind2 and n == m ):
                        H_Total[index_total_1, index_total_2] += OMEGA * n

                    # Interaction Term:
                        # Off-diagonal in photon: Fock Basis
                        # Diagonal in matter, Real-Space Basis
                    if ( R_ind1 == R_ind2 ):
                        CHI_mat = op_b.T + op_b
                        photonic_shift_magnitude = X_im * CHI_mat[n,m]
                        index_shift =  int(photonic_shift_magnitude // dR) ### Estimate real-space shift in index units
                        print(f"Index Shift, Displacement, dR: {index_shift}, {np.round(photonic_shift_magnitude,4)}, {np.round(dR,4)}")
                        if ( R_ind1 + index_shift >= len(RGrid) ):
                            #index_shift -= len(RGrid) # IS THIS CORRECT TO DO ??? I GUESS NOT. 
                            continue # SHOULD SET TO ZERO SHIFT ? I GUESS NOT.
                        H_Total[ index_total_1, index_total_2 ] += Vx_1D[R_ind1 + index_shift]

    return H_Total

def get_H_Total_Fock_Real__Pauli_Fierz( Had, MU, op_b, A0 ):

    H_Total = np.zeros(( len(Had)*Nf, len(Had)*Nf ) ) # Pauli-Fierz is real-valued in this case.

    I_m = np.identity( len(Had) )
    I_p = np.identity( Nf )
    
    H_ph    = np.diag( np.arange(Nf) * wc ) # Photonic Hamiltonian
    H_Total =  kron( Had, I_p )             # Electronic Hamiltonian

    ### Interaction Hamiltonian ###
    H_Total += kron( I_m, H_ph )
    H_Total += kron( MU, op_b.T + op_b ) * (A0 * wc) # chi = A0 * wc
    H_Total += kron( MU @ MU, I_p ) * (A0 * wc)**2 / wc

    return H_Total

def get_H_Total_PFS( Had, MU, A0 ):
    """
    Input: 
        Had, matter hamiltonian (diagonal) energies from electronic structure
        MU,   matter transition dipole matrix from electronic structure
    Output: Hpol, Pauli-Fierz Hamiltonian with single-mode cavity in PFS basis
    """

    def getSmn( m,n,qcm,qcn ):
       
        def fock_overlap_analytic( m,n,qcm,qcn ):

            dZP = np.sqrt( 1/(2*wc) )
            qc0 = qcn - qcm
            ξ   = qc0/(4.0 * dZP)
            G   = np.exp(-2*ξ**2.0)
            X   = (2.0*ξ)**2
            if ( m <= n ):
                S_mn = G * (-ξ * 2)**(n-m) * np.sqrt(fact(m)/fact(n)) * Lg(X,m,n-m)  # n < m
            if ( m > n ):
                S_mn = G * (ξ * 2)**(m-n) * np.sqrt(fact(n)/fact(m)) * Lg(X,n,m-n)  # n < m
            return S_mn


        S_mn = fock_overlap_analytic(m,n,qcm,qcn)           # Analytic Overlap of Fock a and Fock b

        return S_mn

    def HO(x,w,n): # Gets real-space harmonic oscillator function
        cons1 = 1.0/((2.0**n) * fact(n))**(1/2)
        cons2 = ( w / np.pi )**(1/4)
        exp  = np.exp(- (w*(x**2)/2.0)) 
        hermit = Hm(n, np.sqrt(w) * x)
        F_x = cons1 * cons2 * exp * hermit
        return F_x

    def qc(MUii):
        const = np.sqrt( 2.0/wc**3.0 )
        return -wc * A0 * MUii * const

    Hpol = np.zeros((NTrunc*Nf, NTrunc*Nf))

    Had = Had[:NTrunc,:NTrunc]
    MU = MU[:NTrunc,:NTrunc]

    ##### Transform Matter Hamiltonian to MH Basis #####
    MUii, UMU = np.linalg.eigh(MU)
    H_MH = UMU.T @ Had @ UMU


    """
    # Check and enforce some phase on the MH basis
    print( A0, UMU[:,0] )
    fix_phase = np.zeros(( NTrunc, NTrunc ))
    for j in range( NTrunc ):
        for k in range( NTrunc ):
            if ( j == k ):
                fix_phase[j,k] = 1.0 # Keep diagonal same sign
            else:
                fix_phase[j,k] = -1.0 # Switch sign of off-diagonal
    if ( H_MH[0,1] > 0 ): # Choose to enforce some phase
        H_MH = np.einsum( "ab,ab->ab", H_MH[:,:], fix_phase[:,:] )
    """

    #print("Truncated Matrix Dimension:", NTrunc * Nf)

    # Get state-dependent Fock shift
    #qc_all = np.zeros(( NTrunc * Nf ))
    #for a in range( NTrunc ):
    #    qc_all[a] = qc(MUii[a])

    for a in range( NTrunc ): # Matter
        for m in range( Nf ): # PFS
            for b in range( NTrunc ): # Matter
                for n in range( Nf ): # PFS
                    polIND1 = a * Nf + m
                    polIND2 = b * Nf + n

                    if ( a==b and m==n ):
                        Hpol[polIND1,polIND2]  += wc * (m + 0.50000)      # Energy of Fock State
                        Hpol[polIND1,polIND2]  += H_MH[a,a]              # Energy of Matter MH State
                    
                    if ( a != b ):
                        qcm = qc(MUii[a])
                        qcn = qc(MUii[b])
                        S_mn = getSmn( m,n,qcm,qcn )                      # Overlap of Fock m and Fock n
                        #print( a,b,m,n,round(H_MH[a,b],4),round(S_mn,4) )
                        Hpol[polIND1,polIND2] += H_MH[a,b] * S_mn         # Matter (MH) Interaction Weighted by Fock Overlap

    return Hpol




def get_H_Total_Fock_Real__Pauli_Fierz__Truncated( Had, MU, op_b, A0 ): # Enforce additional matter trunction

    print(f"(Nf, Nm): ({Nf}, {NTrunc})")

    H_Total = np.zeros(( NTrunc*Nf, NTrunc*Nf ) ) # Pauli-Fierz is real-valued in this case.

    I_m = np.identity( NTrunc )
    I_p = np.identity( Nf )
    
    H_ph    = np.diag( np.arange(Nf) * wc ) # Photonic Hamiltonian
    
    # Diagonal Pieces
    H_Total += kron( Had[:NTrunc,:NTrunc], I_p )             # Electronic Hamiltonian
    H_Total += kron( I_m, H_ph )                            # Photonic Hamiltonian
    
    ### Interaction Hamiltonian ###
    H_Total += kron( MU[:NTrunc,:NTrunc], op_b.T + op_b ) * (A0 * wc) # chi = A0 * wc
    H_Total += kron( MU[:NTrunc,:NTrunc] @ MU[:NTrunc,:NTrunc], I_p ) * wc * A0**2

    return H_Total

def get_H_Total_Fock_Real__Pauli_Fierz__Wrapper( g_wc ):

    def compute_photon_number_PF_basis( N_PF, U, op_b, I_m ):
        op_b_Total = kron( I_m, op_b.T @ op_b )
        for j in range( len(N_PF) ):
            N_PF[j] = U[:,j].T @ op_b_Total @ U[:,j] 
        return N_PF

    def compute_photon_number_PF_in_pA_basis( N_PF_in_pA, U, op_b, I_m, I_ph, A0, MU ):


        adaga_PF_in_pA = kron( I_m, op_b.T @ op_b ) + kron( MU, op_b.T + op_b) * A0 + kron( MU @ MU, I_ph ) * A0**2 - 0.5

        for j in range( len(N_PF_in_pA) ):
            N_PF_in_pA[j] = U[:,j].T @ adaga_PF_in_pA @ U[:,j]
        return N_PF_in_pA

    print(f"g_wc = {np.round(g_wc,7)}")

    A0 = (g_wc * wc) / qe / np.sqrt( wc / m0 ) #* 27.2114 * 10

    print(f"A0 = {np.round(A0,7)}")


    _, _, MU, Had = get_Real_space_data() # Need electronic dipoles for interactions with photon field
    #H_Total_FockBasis = get_H_Total_Fock_Real__Pauli_Fierz( Had, MU, get_b() )
    H_Total_FockBasis = get_H_Total_Fock_Real__Pauli_Fierz__Truncated( Had, MU, get_b(), A0 )
    #H_Total_FockBasis = get_H_Total_Fock_Real__Pauli_Fierz( Had, MU, get_b(), A0 )
    #plot_H( H_Total_FockBasis, name="H_Total_FockBasis.jpg" )
    E, U = np.linalg.eigh( H_Total_FockBasis )

    print (f"\tFinished g_wc = {g_wc}")

    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", E )
    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat", E - E[0] )
    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
    #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    ### COMPUTE VARIOUS OBSERVABLES ###
    Nmatter = len(E)//Nf # Accounts for possible truncation of matter states

    N_PF = np.zeros(( len(E) ))
    N_PF = compute_photon_number_PF_basis( N_PF, U, get_b(), np.identity( Nmatter ) )
    np.savetxt( f"{DATA_DIR}/N_PFBasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", N_PF )

    N_PF_in_pA = np.zeros(( len(E) ))
    N_PF_in_pA = compute_photon_number_PF_in_pA_basis( N_PF_in_pA, U, get_b(), np.identity( Nmatter ), np.identity( Nf ), A0, MU[:Nmatter,:Nmatter] )
    np.savetxt( f"{DATA_DIR}/N_PF_in_pABasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", N_PF_in_pA )


def get_H_Total_PFS_wrapper( g_wc ):


    A0 = (g_wc * wc) / qe / np.sqrt( wc / m0 )
    print("PFS g_wc (A0) = {:2.3f} ({:2.3f})".format(g_wc, A0 ))


    _, _, MU, Had = get_Real_space_data() # Need electronic dipoles for interactions with photon field
    H_Total_FockBasis = get_H_Total_PFS( Had, MU, A0 )
    E, U = np.linalg.eigh( H_Total_FockBasis )

    print (f"\tFinished g_wc = {g_wc}")

    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_NFock{Nf}_NM{NTrunc}.dat", E )
    np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_NFock{Nf}_NM{NTrunc}_Transition.dat", E - E[0] )
    #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    ### COMPUTE VARIOUS OBSERVABLES ###
    Nmatter = len(E)//Nf # Accounts for possible truncation of matter states


def get_Reciprocal_space_data():
    VMat_k = np.loadtxt( "Vx/VMat_k.dat", dtype=complex )
    KGrid  = np.loadtxt( "Vx/KGrid.dat" )
    return KGrid, VMat_k

def get_Real_space_data():
    NR = 2048
    MU = np.loadtxt( f"Vx_NR{NR}/MU.dat" ) # Electronic Dipole Matrix, NxN
    tmp  = np.loadtxt( f"Vx_NR{NR}/Vx.dat" ) # Real-space grid, Real-space Potential
    RGrid, Vx_1D  = tmp[:,0], tmp[:,1]
    Had = np.diag( np.loadtxt(f"Vx_NR{NR}/Ex.dat") ) # Loads exact eigenenergies
    #Had = np.diag( np.loadtxt("Vx/Ex_Transition.dat.dat") ) # Loads eigenenergies with E0 = 0
    return Vx_1D, RGrid, MU, Had

def get_Pc_Grid(): # Don't use this. Not practical.
    pc_Grid = np.linspace( -10,10,Npc )
    #pc_Grid = np.arange( -10,10,0.1 )
    return pc_Grid

def plot_H( H_Total, name="H_Total.jpg" ):
    EZero_vec = np.ones(( len(H_Total) )) * H_Total[0,0]
    plt.contourf(  np.log( np.abs( H_Total - np.diag( H_Total[np.diag_indices(len(H_Total))] - EZero_vec ) ) + 0.01 ) )
    plt.colorbar()
    plt.title( "Log[|H|]", fontsize=15)
    plt.savefig("data/H_Total.jpg", dpi=500)
    plt.clf()

def main_serial():

    get_Globals()

    if ( HAM == 'AD' and BASIS_PHOTON == "Pc" and BASIS_ELECTRON == "K" ): #### SOVE IN Pc BASIS ####
        # Get matter grid (K) and potential (Vk)
        KGrid, VMat_k = get_Reciprocal_space_data() # Returns basis for momentum space as well as potential in complex momentum space V(k)
        pc_Grid = get_Pc_Grid() # Get photonic grid: pc
        print( f"Full Size of matrix (without truncation): {len(KGrid)}*{len(pc_Grid)} = {len(KGrid)*len(pc_Grid)}" )
        H_Total_PcBasis = get_H_Total_PcBasis( KGrid, pc_Grid, VMat_k )
        #plot_H( H_Total_PcBasis, name="H_Total_PcBasis.jpg" )
        print(f"Diagonalizing Hamiltonian.\n")
        E, U = np.linalg.eigh( H_Total_PcBasis )

        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    elif( HAM == 'AD' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "K" ): #### SOVE IN FOCK BASIS ####
        # Get matter grid (K) and potential (Vk)
        KGrid, VMat_k = get_Reciprocal_space_data() # Returns basis for momentum space as well as potential in complex momentum space V(k)
        print( f"Full Size of matrix (without truncation): {len(KGrid)}*{Nf} = {len(KGrid)*Nf}" )
        ###H_Total_FockBasis = get_H_Total_Fock_Reciprocal__PI_SHIFT_FFT_CONVOLUTION( KGrid, VMat_k ) # DO NOT USE. WRONG CONVOLUTION.
        H_Total_FockBasis = get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION( KGrid, VMat_k, get_b() )
        #plot_H( H_Total_FockBasis, name="H_Total_FockBasis.jpg" )
        print(f"Diagonalizing Hamiltonian.\n")
        E, U = np.linalg.eigh( H_Total_FockBasis )

        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    #### NOT YET WORKING ####
    elif( HAM == 'AD' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "R" ): #### SOVE IN FOCK BASIS ####
        Vx_1D, RGrid, _, _ = get_Real_space_data() # Do not need MU to solve asym. decoupled Hamiltonian
        print( f"Full Size of matrix (without truncation): {len(RGrid)}*{Nf} = {len(RGrid)*Nf}" )
        H_Total_FockBasis = get_H_Total_Fock_Real__CHI_SHIFT( RGrid, Vx_1D, get_b(), T_el( RGrid ) )
        #plot_H( H_Total_FockBasis, name="H_Total_FockBasis.jpg" )
        print(f"Diagonalizing Hamiltonian.")
        E, U = np.linalg.eigh( H_Total_FockBasis )

        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

    elif( HAM == 'PF' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "R" ): #### SOVE IN FOCK BASIS ####
        _, _, MU, Had = get_Real_space_data() # Need electronic dipoles for interactions with photon field
        print( f"Full Size of matrix (without truncation): {len(Had)}*{Nf} = {len(Had)*Nf}" )
        #H_Total_FockBasis = get_H_Total_Fock_Real__Pauli_Fierz( Had, MU, get_b() )
        H_Total_FockBasis = get_H_Total_Fock_Real__Pauli_Fierz__Truncated( Had, MU, get_b() )
        #plot_H( H_Total_FockBasis, name="H_Total_FockBasis.jpg" )
        print(f"Diagonalizing Hamiltonian.\n")
        E, U = np.linalg.eigh( H_Total_FockBasis )

        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )


    elif( HAM == 'PFS' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "R" ):
        _, _, MU, Had = get_Real_space_data() # Need electronic dipoles for interactions with photon field
        print( f"Full Size of matrix (without truncation): {len(Had)}*{Nf} = {len(Had)*Nf}" )
        H_Total = get_H_Total_PFS( Had, MU, 0.1 )
        print(f"Diagonalizing Hamiltonian.\n")
        E, U = np.linalg.eigh( H_Total )

        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}.dat", E )
        np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition.dat", E - E[0] )
        #np.savetxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_A0{A0}_wc{wc}_Transition_NORM.dat", (E-E[0])/(E[1]-E[0]) )
        #np.savetxt( f"U_{BASIS_PHOTON}_{BASIS_ELECTRON}.dat", U )

def main_parallel():

    get_Globals()
    
    g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), 96 ) )
    #g_wc_list = np.linspace( 0, 3, 96 )
    with mp.Pool(processes=NCPUS) as pool:
        if( HAM == 'AD' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "K" ): #### SOVE IN FOCK BASIS ####
            #pool.map( get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION__Wrapper, A0_list )
            pool.map( get_H_Total_Fock_Reciprocal__CHI_SHIFT__FFT_CONVOLUTION__Wrapper, g_wc_list )
        elif( HAM == 'PF' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "R" ): #### SOVE IN FOCK BASIS ####
            pool.map( get_H_Total_Fock_Real__Pauli_Fierz__Wrapper, g_wc_list )
        elif( HAM == 'PFS' and BASIS_PHOTON == "Fock" and BASIS_ELECTRON == "R" ): #### SOVE IN FOCK BASIS ####
            pool.map( get_H_Total_PFS_wrapper, g_wc_list )
if ( __name__ == '__main__' ):
    #main_serial()
    main_parallel()

