import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

ng = 48
wc = 0.986 # a.u.
nf = 5
nR = 64
BASIS = "AD"

g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), ng ))


DIR = "plot_data/"
sp.call(f"mkdir -p {DIR}", shell=True)
#######
NPol = nf * nR

EPol = np.zeros(( len(g_wc_list), NPol ))
for A0_IND, A0 in enumerate( g_wc_list ):
    print (A0)
    #EPol[A0_IND,:] = np.loadtxt( f"data/E_{BASIS}_A0{np.round(A0,1)}_wc{wc}.dat" ) # Extract actual energies
    EPol[A0_IND,:] = np.loadtxt( f"data/E_{BASIS}_{nf}_{nR}_gwc{np.round(A0,7)}_wc{wc}_Transition.dat" ) # Extract transition energies

#plot_states = [0,1,2,3,4,5,6,7,8,9]
plot_states = np.arange( 10 )
for state in plot_states:
    plt.loglog( g_wc_list, EPol[:,state], "-", label=f"p{state}" )
    #plt.plot( g_wc_list, EPol[:,state], label=f"p{state}" )
    #plt.scatter( g_wc_list, EPol[:,state] )
plt.legend()
plt.xlim(1e-2,100)
plt.ylim(0.05,3.5)
plt.xlabel("$A_0$ (a.u.)",fontsize=15)
plt.ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DIR}/A0_Scan.jpg",dpi=300)


