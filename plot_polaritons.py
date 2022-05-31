import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

#g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), 96 ) )
g_wc_list = np.linspace( 0, 3, 96 )

wc = 0.98600303 #* 27.2117 # a.u.
#wc = 1.03202434509 #* 27.2117 # a.u.

Nph = 40
Nmatter  = 64
HAM = "AD"
BASIS_PHOTON = "Fock"
BASIS_ELECTRON = "K"
DATA_DIR = f"data_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}"
#PLOT_DIR = f"plot_data_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}/"

#######
#sp.call(f"mkdir -p {PLOT_DIR}", shell=True)
NPol = Nph * Nmatter

N_PF = np.zeros(( len(g_wc_list), NPol ))
EPol = np.zeros(( len(g_wc_list), NPol ))
XI = np.zeros(( len(g_wc_list)  ))
g_wc = np.zeros(( len(g_wc_list)  ))

for g_wc_IND, g_wc in enumerate( g_wc_list ):
    print (g_wc)
    #try:
        #EPol[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{g_wc}_wc{wc}.dat" ) # Extract actual energies
    EPol[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/E_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}_Transition.dat" ) # Extract transition energies
    #N_PF[g_wc_IND,:] = np.loadtxt( f"{DATA_DIR}/N_PFBasis_{HAM}_{BASIS_PHOTON}_{BASIS_ELECTRON}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat" ) # Extract transition energies
        #XI[g_wc_IND] =   np.loadtxt( f"{DATA_DIR}/XI_g_data_{g_wc}.dat" )[1] # Extract transition energies
        #g_wc[g_wc_IND] = np.loadtxt( f"{DATA_DIR}/XI_g_data_{g_wc}.dat" )[0] # Extract transition energies
    #except:
    #    continue
#plot_states = [0,1,2,3,4,5,6,7,8,9]
plot_states = np.arange( 40 )
for state in plot_states:
    #print( g_wc_list )
    plt.loglog( g_wc_list, EPol[:,state], "-o", label=f"p{state}" )
    #plt.plot( g_wc_list, EPol[:,state], label=f"p{state}" )
    #plt.scatter( g_wc_list, EPol[:,state] )
plt.legend()
plt.xlim(0.01,100)
#plt.xlim(1,10**5)
plt.ylim(0.05,3.6)
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
plot_states = np.arange( 20 )
for state in plot_states:
    #if ( np.max(N_PF[:,state]) ):
    #    for k in range( len(g_wc_list) ):
    #        if ( N_PF[k,state] > 5 ):
    #            N_PF[k,state] = 0
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}" )
    #plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}", c=N_PF[:,state], s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=5 )
    #plt.scatter( g_wc_list, EPol[:,state], c=N_PF[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=10 )
    plt.scatter( g_wc_list, EPol[:,state], c=N_PF[:,state],s=70,marker="o",edgecolor='none',cmap=plt.cm.hsv, vmin=0, vmax=3 )
plt.legend()
plt.colorbar()
plt.xlim(0.01,2.5)
#plt.xlim(1,10**5)
plt.ylim(0.05,3.5)
#plt.ylim(1e-6,3.5)
plt.xlabel("$g/\omega_c$ (a.u.)",fontsize=15)
plt.ylabel("Average Photon Number (a.u.)",fontsize=15)
plt.savefig(f"{DATA_DIR}/photon_number_PFBasis.jpg",dpi=300)
plt.clf()






