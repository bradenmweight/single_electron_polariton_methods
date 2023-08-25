import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

ng = 128
wc = 1 # a.u.
nf = 5
n_kappa = 101
BASIS = "RAD"
a_0 = 4
nk = 32
dark = False

# file_location = "/home/mtayl29/single_electron_polariton_methods/erf-potential-modes/"
file_location = "./"

g_wc_list = np.exp( np.linspace( np.log(10**-2), np.log(100), ng ))
k_list = np.linspace(0, np.pi / a_0, nk)

DIR = "plot_data/"
sp.call(f"mkdir -p {DIR}", shell=True)

NPol = nf * n_kappa

try:
     EPol = np.load(f"{DIR}/plot_data.npy")
except:
    EPol = np.zeros(( len(g_wc_list),len(k_list), NPol ))

    zpe = np.min(np.loadtxt( f"{file_location}data1/E_{BASIS}_k{np.round(0.0,3)}_{nf}_{n_kappa}_gwc{np.round(100.0,7)}_wc{np.round(wc,4)}.dat" ))

    for A0_IND, A0 in enumerate( g_wc_list ):
        for k_ind, k in enumerate(k_list):
            print (A0)
            EPol[A0_IND,k_ind,:] = np.loadtxt( f"{file_location}data1/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(A0,7)}_wc{np.round(wc,4)}.dat" ) -zpe # Extract energies

    print(zpe)

colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=nk)

numbers = (np.linspace(0,1,nk))
kcolors = ["%.2f" % number for number in numbers]
label = 'bright'
cmap = cm

plot_states = np.arange( 40 )

if dark:
    plt.style.use('dark_background')
    kcolors = np.flip(kcolors)
    label = 'dark'
    # cmap = 'gist_gray_r'
    cmap = 'Reds_r'


fig, ax = plt.subplots()

all_segments = np.zeros(( len(k_list) * len(plot_states),len(g_wc_list),2))

for ijk in range(len(plot_states)):
    for lmn in range(len(k_list)):
        all_segments[ijk + (len(k_list) - lmn - 1)*len(plot_states),:,1] = EPol[:,lmn,ijk]
        all_segments[ijk + (len(k_list) - lmn - 1)*len(plot_states),:,0] = g_wc_list

# cols = np.flip(k_list)
# for state in plot_states:
#     cols = np.concatenate((cols,np.flip(k_list)))

cols = np.ones((len(plot_states))) * np.max(k_list)
for k in np.flip(range(len(k_list) - 1)):
    cols = np.concatenate((cols, np.ones((len(plot_states))) * k_list[k]))

plt.subplots_adjust(left=0.125,
                    bottom=0.2,
                    right=0.9,
                    top=0.92,
                    wspace=0.2,
                    hspace=0.2)

fs = 15
plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-2,100)
# plt.xlim(10,100)
plt.ylim(0.01,7)
# plt.ylim(0.0,0.4)
plt.xlabel("$g_0 / \omega_0$",fontsize=fs)
plt.ylabel("Energy (a.u.)",fontsize=fs)

lc = LineCollection(all_segments, cmap= cmap)
lc.set_array(cols)
lc.set_linewidth(1.9)
line = ax.add_collection(lc)

# cbar = fig.colorbar(line,ax=ax, shrink = 0.6, 
#                     orientation = 'horizontal', 
#                     location = 'bottom', 
#                     pad = -0.25)
# # cbar.ax.tick_params(labelsize=fs)
# # cbar.ax.set_yticklabels(["0", ""])
# cbar.set_label('k',fontsize=13)

plt.savefig(f"{DIR}/A0_Scan_{label}_lc.jpg",dpi=600)
plt.savefig(f"{DIR}/A0_Scan_{label}_lc.svg",format='svg')

np.save(f"{DIR}/plot_data.npy", EPol)

# output = np.zeros(( len(g_wc_list), NPol + 1 ))
# output[:,0] = g_wc_list
# output[:,1:] = EPol
# np.savetxt( f"{DIR}/plot_data.dat", output)

