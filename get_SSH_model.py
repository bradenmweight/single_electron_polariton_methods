import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
import subprocess as sp



def get_Globals():
    global NSites

    NSites = 100

    sp.call("mkdir -p Vx", shell=True)

def get_uniform_SSH_noPBC():
    Ham = np.zeros(( NSites, NSites ))
    lam  = 0.1
    column = [1.0, lam]
    for j in range( NSites-2 ):
        column.append( 0.0 )
    Ham[:,:] = toeplitz( column )
    E, U = np.linalg.eigh( Ham )
    
    return Ham, E, U

def get_Ham_K( Ham_s ):

    def get_FFT_2D( f_x, dx ):
        k = np.fft.fftfreq( NSites ) * ( 2 * np.pi / dx )
        f_k = np.fft.fftn( f_x, norm="ortho" ) / NSites
        return k, f_k

    def get_FFT_2D_BRADEN( f_x, dx ):
        
        f_k = np.zeros(( NSites, NSites ), dtype=complex)
        KGrid = 2 * np.pi / dx * np.arange( NSites )
        #for k1 in range( NSites ):
            #print(k1)
            #for k2 in range( NSites ):
                #for x1 in range( NSites ):
                #    for x2 in range( NSites ):
                #        kdiff = abs(KGrid[k1] - KGrid[k2])
                #        rdiff = abs(x1 - x2)
                #        phase = 1.0j * kdiff * rdiff
                #        f_k[k1,k2] += np.exp( phase ) * f_x[ x1, x2 ]
        kdiff = np.outer( KGrid, KGrid )
        rdiff = np.outer( np.arange( NSites ), np.arange( NSites ) )
        phase = np.einsum( "nm,jk->nmjk", 1j*kdiff, rdiff )
        f_k[:,:] = np.einsum( "nmjk,jk->nm", np.exp( phase )[:,:,:,:], f_x[:,:] )
        return KGrid, f_k #/ ( 2 * np.pi * NSites )

    #return get_FFT_2D( Ham_s, 1.0 )
    return get_FFT_2D_BRADEN( Ham_s, 1.0 )

def main():
    get_Globals()

    # Get Hamiltonian in site basis -- Includes both 1e potential and kinetic terms
    Ham_s, E, U = get_uniform_SSH_noPBC()
    
    KGrid, Ham_K = get_Ham_K( Ham_s )
    #print(Ham_s)
    #print(Ham_K)
    print( np.linalg.eigh(Ham_s)[0][:5] )
    print( np.linalg.eigh(Ham_K)[0][:5] )
    print( np.linalg.eigh(Ham_K)[0][:5] / np.linalg.eigh(Ham_s)[0][:5]  )

    exit()


    plot_Vk( KGrid, Vk )
    save_data( RGrid, Vx, KGrid, Vk )

    if ( SOLVE_X ):
        E,U = get_solutions_X( RGrid, Vx )
        plot_wfns_X( E,U,Vx,RGrid )
        get_dipole_matrix( RGrid, U )
    if ( SOLVE_K ):
        E,U = get_solutions_K( KGrid, Vk )
        #plot_wfns_K( E,U,Vx,RGrid )

    print( "\n\tAll done. Have a nice day! :)\n" )

if ( __name__ == '__main__' ):
    main()





