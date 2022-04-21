#!/home/mtayl29/anaconda3/bin/python
import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from scipy.special import erf
import get_vx 
#from numba import jit

# def get_Globals():
#     global nR

#     nR = 32
#     sp.call("mkdir -p Vx", shell=True)

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
    return E,U, VMat_k

def plot_Vk( KGrid, Vk ):
    plt.plot( KGrid, np.log( np.abs(np.real(Vk)) + 0.001 ), label="RE" )
    plt.plot( KGrid, np.log( np.abs(np.imag(Vk)) + 0.001), label="IM" )
    plt.ylim(-7,8)
    plt.legend()
    plt.savefig("Vx/Vk.jpg", dpi=300)
    plt.clf()

def plot_Vx( RGrid, Vx ):
    plt.plot( RGrid, Vx )
    plt.savefig("Vx/Vx.jpg", dpi=300)
    plt.clf()

def get_V(nR):
    sp.call("mkdir -p Vx", shell=True)

    # RGrid, Vx = get_vx.get_QHO(nR)
    RGrid, Vx = get_vx.get_double_well(nR)
    plot_Vx( RGrid, Vx )

    KGrid, Vk = get_V_k( RGrid, Vx )
    plot_Vk( KGrid, Vk )

    save_data( RGrid, Vx, KGrid, Vk )

    E, U, Vmat_k = get_solutions_K( KGrid, Vk )
    #plot_wfns_K( E,U,Vx,RGrid )

    wc = E[1] - E[0] #Resonant Condition

    return Vk, wc, KGrid

if ( __name__ == '__main__' ):
    get_V(512)

