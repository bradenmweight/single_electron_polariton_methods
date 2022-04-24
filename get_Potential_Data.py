import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from scipy.special import erf
#from numba import jit



def get_Globals():
    global nR, FIX_BOUNDARIES, SOLVE_X, SOLVE_K

    nR = 512
    FIX_BOUNDARIES = False # Adds high wall potential to boundary of box.
    SOLVE_X = True
    SOLVE_K = True

    sp.call("mkdir -p Vx", shell=True)

def get_V_x__201Erfs( RGrid ):
    a0 = 20 # Lattice Spacing for Erf(x)/x, Erf( x-a0 ) / (x-a0), a0 = Length (a.u.)
    r0 = 5  # Erf width --> Erf( x/r0 ) / x, r0 = Length (a.u.)
    Vx = np.zeros(( nR ))
    for n in np.arange( -100,101 ):
        Vx += -erf( (RGrid-n*a0) / r0 ) / (RGrid - n*a0)
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )
    return Vx

def get_V_x__41Erfs( RGrid ):
    a0 = 20 # Lattice Spacing for Erf(x)/x, Erf( x-a0 ) / (x-a0), a0 = Length (a.u.)
    r0 = 5  # Erf width --> Erf( x/r0 ) / x, r0 = Length (a.u.)
    Vx = np.zeros(( nR ))
    for n in np.arange( -20,21 ):
        Vx += -erf( (RGrid-2*n*a0) / r0 ) / (RGrid - 2*n*a0)
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )
    return Vx

def get_V_x__1Erfs( RGrid ):
    a0 = 20 # Lattice Spacing for Erf(x)/x, Erf( x-a0 ) / (x-a0), a0 = Length (a.u.)
    r0 = 5  # Erf width --> Erf( x/r0 ) / x, r0 = Length (a.u.)
    Vx = np.zeros(( nR ))
    Vx += -erf( RGrid / r0 ) / RGrid
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )
    return Vx

def get_Demler_FIG3_a():
    RGrid = np.linspace(-2.6, 2.6, nR) # Seems like 512 points is best with these parameters
    Vx = np.zeros(( nR ))
    lam = 50
    mu  = 95
    Vx += -lam * RGrid**2 / 2 + mu * RGrid**4 / 4
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )

    Vx -= np.min( Vx )

    return RGrid, Vx

def get_Demler_FIG3_b( RGrid ):
    Vx = np.zeros(( nR ))
    lam = 3
    mu  = 3.85
    Vx += -lam * RGrid**2 / 2 + mu * RGrid**4 / 4
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )
    return Vx

def get_QHO():
    RGrid = np.linspace(-5,5,nR)
    Vx = np.zeros(( nR ))
    w = 1.0
    Vx += 0.5 * w**2 * RGrid**2
    if ( FIX_BOUNDARIES ):
        Vx = correct_Boundaries( Vx )
    #Vx -= np.min( Vx ) # Shift MIN(V) to zero.
    return RGrid, Vx

def correct_Boundaries(Vx):
    Vx[-10:] = np.ones(( 10 )) * 5
    Vx[:10] = np.ones(( 10 )) * 5
    return Vx

def plot_Vx( RGrid, Vx ):
    plt.plot( RGrid, Vx )
    plt.savefig("Vx/Vx.jpg", dpi=300)
    plt.clf()

def plot_Vk( KGrid, Vk ):
    plt.plot( KGrid, np.log( np.abs(np.real(Vk)) + 0.001 ), label="RE" )
    plt.plot( KGrid, np.log( np.abs(np.imag(Vk)) + 0.001), label="IM" )
    plt.ylim(-3,8)
    plt.legend()
    plt.savefig("Vx/Vk.jpg", dpi=300)
    plt.clf()

def get_V_k( RGrid, Vx ):
    def get_FFT( f_x, dx):
        k = np.fft.fftfreq(len(f_x)) * (2.0 * np.pi / dx)
        f_k = np.fft.fft( f_x, norm="ortho" ) / np.sqrt(len(f_x))
        return k, f_k
    return get_FFT( Vx, RGrid[1] - RGrid[0] )

def save_data( RGrid, Vx, KGrid, Vk ):
    np.savetxt( "Vx/Vx.dat", np.c_[ RGrid, Vx ] ) # Real-Valued
    np.savetxt( "Vx/Vk.dat", np.c_[ KGrid, Vk ] ) # Complex-Valued

    tmp = np.loadtxt( "Vx/Vk.dat", dtype=complex )
    #assert ( tmp[50,1] == Vk[50] ),  "Saved data is not the same as the original data..." 

    # Save VMat_k as matrix
    VMat_k = get_VMat_k( KGrid, Vk )
    np.savetxt( "Vx/KGrid.dat", KGrid )
    np.savetxt( "Vx/VMat_k.dat", VMat_k )
    VMat_k = np.real( VMat_k )
    #plt.contourf( KGrid, KGrid, VMat_k )
    plt.imshow( np.log(np.abs(VMat_k) + 0.001), origin='lower' )
    plt.colorbar()
    plt.savefig("Vx/VMat_k.jpg",dpi=400)
    plt.clf()


def get_solutions_X( RGrid, Vx ):
    def get_T( RGrid ): # DVR Basis
        dR = RGrid[1] - RGrid[0]
        T = np.zeros(( nR, nR ))
        for n in range( nR ):
            for m in range( nR ):
                norm = (-1) ** (n-m) / dR**2
                if ( n == m ):
                    T[n,m] = np.pi**2 / 3 
                else:
                    T[n,m] = 2 / (n-m)**2
                T[n,m] *= norm
        return  T/2
    
    print( f"Solving ({len(RGrid)}, {len(RGrid)}) Hamiltonian in X-space." )
    E,U = np.linalg.eigh( np.diag(Vx) + get_T( RGrid ) )
    np.savetxt( f"Vx/Ex.dat", E )
    np.savetxt( f"Vx/Ex_Transition.dat", E - E[0] )
    np.savetxt( f"Vx/Ex_Transition_NORM.dat", (E - E[0]) / ( E[1] - E[0] ) )
    
    #np.savetxt( f"Vx/U_x.dat", U )
    return E,U

def get_VMat_k( KGrid, Vk ):
    VMat_k = np.zeros(( len(KGrid), len(KGrid) ), dtype=complex)
    for n in range( len(KGrid) ): # k
        for m in range( len(KGrid) ): # k'
            #if ( n != m ):
            VMat_k[n,m] = Vk[ n-m ]
    return VMat_k

def get_solutions_K( KGrid, Vk ):
    
    VMat_k = get_VMat_k( KGrid, Vk )

    print( f"Solving ({len(KGrid)}, {len(KGrid)}) Hamiltonian in K-space." )
    E,U = np.linalg.eigh( VMat_k + np.diag( KGrid ** 2 / 2 ) )
    np.savetxt( f"Vx/Ek.dat", E )
    np.savetxt( f"Vx/Ek_Transition.dat", E - E[0] )
    np.savetxt( f"Vx/Ek_Transition_NORM.dat", (E - E[0]) / ( E[1] - E[0] ) )
    
    #np.savetxt( f"Vx/U_k.dat", U )
    return E,U

def plot_wfns_X( E,U,Vx,RGrid ):
    # Plot potential along with states S0 through S3 wavefunctions.
    plt.plot( RGrid, Vx, c="black", alpha=0.5, linewidth=8 )
    plt.plot( RGrid, 10*U[:,0] + E[0], c="red", alpha=1.0, linewidth=3 )
    plt.plot( RGrid, 10*U[:,1] + E[1], c="blue", alpha=1.0, linewidth=3 )
    plt.plot( RGrid, 10*U[:,2] + E[2], c="green", alpha=1.0, linewidth=3 )
    plt.plot( RGrid, 10*U[:,3] + E[3], c="orange", alpha=1.0, linewidth=3 )
    plt.xlim(-2,2)
    plt.ylim(0,10)
    plt.xlabel( "X (a.u.)" ,fontsize=15)
    plt.ylabel( "Energy / Wavefunction (a.u.)" ,fontsize=15)
    plt.savefig( "Vx/WFN.jpg" ,dpi=300)
    plt.clf()

def get_dipole_matrix( RGrid, U ): # NOT RIGOROUSLY TESTED. NEED TO IMPLEMENT IN K-SPACE AS WELL.
    # This will be somewhat sensitive to the grid spacing,
    #   more-so than the energies themselves.
    dR = RGrid[1] - RGrid[0]
    MU = np.zeros(( len(RGrid), len(RGrid) ))
    for j in range( len(RGrid) ):
        for k in range( len(RGrid) ):
             # Could do better integration. Maybe with scipy.
            MU[ j,k ] = np.sum( U[:,j].T * RGrid * U[:,k] ) * dR
    np.savetxt( "Vx/MU.dat", MU )
    plt.imshow( np.abs(MU), origin='lower' )
    plt.colorbar()
    plt.title("Dipole Matrix",fontsize=15)
    plt.xlabel("State Index",fontsize=15)
    plt.ylabel("State Index",fontsize=15)
    plt.savefig("Vx/MU.jpg",dpi=400)
    plt.clf()

def main():
    get_Globals()

    ###Vx = get_V_x__201Erfs( RGrid )
    ###Vx = get_V_x__41Erfs( RGrid )
    ###Vx = get_V_x__1Erfs( RGrid )
    RGrid, Vx = get_Demler_FIG3_a()
    #Vx = get_Demler_FIG3_b( RGrid )
    #RGrid, Vx = get_QHO()
    
    plot_Vx( RGrid, Vx )
    KGrid, Vk = get_V_k( RGrid, Vx )
    plot_Vk( KGrid, Vk )
    save_data( RGrid, Vx, KGrid, Vk )

    if ( SOLVE_X ):
        E,U = get_solutions_X( RGrid, Vx )
        plot_wfns_X( E,U,Vx,RGrid )
        get_dipole_matrix( RGrid, U )
    if ( SOLVE_K ):
        E,U = get_solutions_K( KGrid, Vk )
        #plot_wfns_K( E,U,Vx,RGrid )


if ( __name__ == '__main__' ):
    main()
