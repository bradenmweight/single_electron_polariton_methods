import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

ng = 128
wc = 1 # a.u.
nf = 5
n_kappa = 101
BASIS = "RAD"
a_0 = 4
nk = 32

g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), ng ))
k_list = np.linspace(0, np.pi / a_0, nk)

DIR = "plot_data/"
sp.call(f"mkdir -p {DIR}", shell=True)
#######
NPol = nf * n_kappa

EPol = np.zeros(( len(g_wc_list),len(k_list), NPol ))

zpe = np.min(np.loadtxt( f"data/E_{BASIS}_k{np.round(0.0,3)}_{nf}_{n_kappa}_gwc{np.round(100.0,7)}_wc{np.round(wc,4)}.dat" ))

for A0_IND, A0 in enumerate( g_wc_list ):
    for k_ind, k in enumerate(k_list):
        print (A0)
        #EPol[A0_IND,:] = np.loadtxt( f"data/E_{BASIS}_A0{np.round(A0,1)}_wc{wc}.dat" ) # Extract actual energies
        EPol[A0_IND,k_ind,:] = np.loadtxt( f"data/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(A0,7)}_wc{np.round(wc,4)}.dat" ) -zpe # Extract energies

print(zpe)

#plot_states = [0,1,2,3,4,5,6,7,8,9]
# kcolors = np.flip(['0.850','0.80','0.750','0.70','0.650','0.60','0.550','0.50','0.450','0.40','0.350','0.30','0.250','0.2','0.15','0.05'])
# kcolors = ['1','0.90','0.850','0.80','0.750','0.70','0.650','0.60','0.550','0.50','0.450','0.40','0.350','0.3','0.25','0.2']

# numbers = np.flip(np.linspace(0.2,1,nk))**2
numbers = (np.linspace(0,1,nk))
kcolors = ["%.2f" % number for number in numbers]

plot_states = np.arange( 40 )
# plt.style.use('dark_background')

for k_ind in np.flip(range(len(k_list))):
    for state in plot_states:
        plt.loglog( g_wc_list, EPol[:,k_ind,state], "-", color=kcolors[k_ind],linewidth = 1.9)
        #plt.plot( g_wc_list, EPol[:,state], label=f"p{state}" )
        #plt.scatter( g_wc_list, EPol[:,state] )
# plt.legend()
plt.xlim(1e-2,100)
plt.ylim(0.01,7)
plt.xlabel("$g / \omega_c$ (a.u.)",fontsize=15)
plt.ylabel("Energy (a.u.)",fontsize=15)
plt.savefig(f"{DIR}/A0_Scan.jpg",dpi=600)
plt.savefig(f"{DIR}/A0_Scan.svg",format='svg')

output = np.zeros(( len(g_wc_list), NPol + 1 ))
output[:,0] = g_wc_list
output[:,1:] = EPol
np.savetxt( f"{DIR}/plot_data.dat", output)

