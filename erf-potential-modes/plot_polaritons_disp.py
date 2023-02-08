import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

nk = 128
wc = 1
nf = 5
n_kappa = 101
BASIS = "RAD"
a_0 = 4
g_wc_array =  [  0,0.1,0.2,0.3,  1, 10,100]
y_max_array = [  5,  5,  5,  5,  5,  1,0.4]

for ijk in range(len(g_wc_array)):
    g_wc = g_wc_array[ijk]
    y_max = y_max_array[ijk]

    k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)


    DIR = "plot_data_disp/"
    sp.call(f"mkdir -p {DIR}", shell=True)
    #######
    NPol = nf * n_kappa

    EPol = np.zeros(( len(k_points), NPol ))
    for k_ind, k in enumerate( k_points ):
        print (k)
        EPol[k_ind,:] = np.loadtxt( f"data/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat" ) + 0.3651387160346118

    # plt.style.use('dark_background')
    fs = 23
    plot_states = np.arange( 505 )
    for state in plot_states:
        plt.plot( k_points, EPol[:,state], "-", color='k')
    plt.ylim(0.0,y_max)
    plt.rcParams["figure.figsize"] = (2,1.5)
    # plt.xlabel("$k$ (a.u.)",fontsize=fs)
    plt.yticks(fontsize = fs)
    plt.xticks(ticks = [- np.pi / 4,0, np.pi / 4], labels = ["$-\pi/a_0$", "0", "$\pi/a_0$"],fontsize = fs)
    # plt.title(f"$g / \omega_c =$ {g_wc}",fontsize=fs)
    # plt.ylabel("Energy (a.u.)",fontsize=fs)
    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}.jpg",dpi=600)
    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}.svg",format='svg')

    output = np.zeros(( len(k_points), NPol + 1 ))
    output[:,0] = k_points
    output[:,1:] = EPol
    np.savetxt( f"{DIR}/plot_data.dat", output)

    plt.clf()

