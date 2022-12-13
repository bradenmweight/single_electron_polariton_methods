import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess as sp

#g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), 96 ) )
g_wc_list = np.linspace( 0, 3, 96 )

wc = 0.98600303 #* 27.2117 # a.u.
#wc = 1.032419 #* 27.2117 # a.u.

Nph = 200
Nmatter  = 50
HAM = "PF"
BASIS_PHOTON = "Fock"
BASIS_ELECTRON = "R"
DATA_DIR = f"data_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}"
#PLOT_DIR = f"plot_data_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}/"

#######
#sp.call(f"mkdir -p {PLOT_DIR}", shell=True)
NPol = Nph * Nmatter

EPol = np.zeros(( len(g_wc_list), NPol ))
N_1 = np.zeros(( len(g_wc_list), NPol )) # Photon number in original basis
N_2 = np.zeros(( len(g_wc_list), NPol )) # Photon number in p.A basis



for g_wc_IND, g_wc in enumerate( g_wc_list ):
    print (g_wc)
    #try:
        #EPol[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{g_wc}_wc{wc}.dat" ) # Extract actual energies
    EPol[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat" ) # Extract transition energies
    N_1[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/N_{HAM}Basis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat" ) # Extract transition energies
    #N_2[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/N_{HAM}_in_pABasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat" ) # Extract transition energies
        #XI[g_wc_IND] =   np.loadtxt( f"{DATA_DIR}/XI_g_data_{g_wc}.dat" )[1] # Extract transition energies
        #g_wc[g_wc_IND] = np.loadtxt( f"{DATA_DIR}/XI_g_data_{g_wc}.dat" )[0] # Extract transition energies
    #except:
    #    continue

# Save Data
np.savetxt(f"{DATA_DIR}/EPOL_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_wc{np.round(wc,4)}_Transition.dat", EPol[:,:100])
np.savetxt(f"{DATA_DIR}/N_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_wc{np.round(wc,4)}_Transition.dat", N_1[:,:100])
#np.savetxt(f"{DATA_DIR}/N_{HAM}inPABasis_{BASIS_PHOTON}_{BASIS_ELECTRON}_wc{np.round(wc,4)}_Transition.dat", N_2[:,:100])

exit()

#plot_states = [0,1,2,3,4,5,6,7,8,9]
plot_states = np.arange( 10 )
for state in plot_states:
    #print( g_wc_list )
    plt.loglog( g_wc_list, EPol[:,state], "-o", label=f"p{state}" )
    #plt.plot( g_wc_list, EPol[:,state], label=f"p{state}" )
    #plt.scatter( g_wc_list, EPol[:,state] )
plt.legend()
plt.xlim(0.01,100)
#plt.xlim(1,10**5)
plt.ylim(0.05,3.5)
#plt.ylim(1e-6,3.5)
plt.xlabel("$g/\omega_c$ (a.u.)",fontsize=15)
plt.ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DATA_DIR}/g_wc_Scan.jpg",dpi=300)
#plt.savefig(f"{PLOT_DIR}/g_wc_Scan.jpg",dpi=300)
plt.clf()



#plt.semilogx( g_wc, XI )
#plt.savefig(f"{DATA_DIR}/XI_g_wc.jpg",dpi=300)


np.savetxt( f"{DATA_DIR}/plot_data_NR{Nmatter}_NF{Nph}_Transition.dat", np.c_[g_wc_list,EPol[:,0],EPol[:,1],EPol[:,2],EPol[:,3],EPol[:,4],EPol[:,5], \
                                                                    EPol[:,6], EPol[:,7], EPol[:,8], EPol[:,9], EPol[:,10], \
                                                                    EPol[:,11], EPol[:,12], EPol[:,13], EPol[:,14], EPol[:,15] ] )


### PLOT PHOTON NUMBER ###

# Original Basis (AD or PF)
"""
plot_states = np.arange( 1,20 )
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for state in plot_states:
    if ( state == 1 ):
        im = plt.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, norm=LogNorm(vmin=0.1, vmax=100) )
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}", c='b', cmap=plt.cm.hsv, vmin=0, vmax=100 )
    ax.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, norm=LogNorm(vmin=0.1, vmax=100) )
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}", c=N_1[:,state], s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=5 )
    #plt.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=100 )
    #plt.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=10 )
plt.legend()
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01,100)
#plt.xlim(0,2.5)
#plt.ylim(-0.05,3.5)
ax.set_ylim(0.05,3.5)
ax.set_xlabel("$g/\omega_c$ (a.u.)",fontsize=15)
ax.set_ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DATA_DIR}/photon_number_{HAM}Basis__NR{Nmatter}_NF{Nph}.jpg",dpi=300)
plt.clf()
"""


plot_states = np.arange( 20 )
for state in plot_states:
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}", c=N_2[:,state], s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=5 )
    plt.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=100 )
    #plt.scatter( g_wc_list, EPol[:,state], c=N_1[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=10 )
plt.legend()
plt.colorbar()
#plt.xlim(0.0,20)
plt.xlim(0,2.5)
plt.ylim(-0.05,3.5)
#plt.ylim(1e-6,3.5)
plt.xlabel("$g/\omega_c$ (a.u.)",fontsize=15)
plt.ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DATA_DIR}/photon_number_{HAM}_in_pABasis__NR{Nmatter}_NF{Nph}.jpg",dpi=300)
plt.clf()


"""
# p.A Basis ("Real" Photon Number)
plot_states = np.arange( 20 )
for state in plot_states:
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}", c=N_2[:,state], s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=5 )
    plt.scatter( g_wc_list, EPol[:,state], c=N_2[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=100 )
    #plt.scatter( g_wc_list, EPol[:,state], c=N_2[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=10 )
plt.legend()
plt.colorbar()
plt.xlim(0.0,20)
#plt.xlim(0,2.5)
plt.ylim(-0.05,3.5)
#plt.ylim(1e-6,3.5)
plt.xlabel("$g/\omega_c$ (a.u.)",fontsize=15)
plt.ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DATA_DIR}/photon_number_{HAM}_in_pABasis__NR{Nmatter}_NF{Nph}.jpg",dpi=300)
plt.clf()

"""
