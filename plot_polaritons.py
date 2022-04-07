import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp


A0_list = []
for j in np.arange( 0.0, 100+0.001, 0.001 ):
    if ( j < 0.01 and ( np.round(j * 1000,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )
    if ( j < 0.1 and j >= 0.01 and ( np.round(j * 100,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )
    if ( j < 1 and j >= 0.1 and ( np.round(j * 10,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )
    if ( j < 10 and j >= 1 and ( np.round(j * 1,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )
    if ( j < 100 and j >= 10 and ( np.round(j * 0.1,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )
    if ( j < 1000 and j >= 100 and ( np.round(j * 0.01,4) ).is_integer() ):
        A0_list.append( np.round(j,4) )

#print( A0_list )

A0_list = np.array( A0_list )


wc = 0.986 # a.u.

Nph = 100
Nk  = 32
BASIS = "Pc"


DIR = "plot_data/"
sp.call(f"mkdir -p {DIR}", shell=True)
#######
NPol = Nph * Nk

EPol = np.zeros(( len(A0_list), NPol ))
for A0_IND, A0 in enumerate( A0_list ):
    print (A0)
    #EPol[A0_IND,:] = np.loadtxt( f"data/E_{BASIS}_A0{np.round(A0,1)}_wc{wc}.dat" ) # Extract actual energies
    EPol[A0_IND,:] = np.loadtxt( f"data/E_{BASIS}_A0{np.round(A0,2)}_wc{wc}_Transition.dat" ) # Extract transition energies

#plot_states = [0,1,2,3,4,5,6,7,8,9]
plot_states = np.arange( 10 )
for state in plot_states:
    plt.loglog( A0_list, EPol[:,state], "-o", label=f"p{state}" )
    #plt.plot( A0_list, EPol[:,state], label=f"p{state}" )
    #plt.scatter( A0_list, EPol[:,state] )
plt.legend()
plt.xlim(0.01,100)
plt.ylim(0.05,6)
plt.xlabel("$A_0$ (a.u.)",fontsize=15)
plt.ylabel("Transition Energy (a.u.)",fontsize=15)
plt.savefig(f"{DIR}/A0_Scan.jpg",dpi=300)


