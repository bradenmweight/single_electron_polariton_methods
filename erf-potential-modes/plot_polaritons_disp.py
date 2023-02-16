import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import subprocess as sp
nk = 128
wc = 1
nf = 5
n_kappa = 101
BASIS = "RAD"
a_0 = 4
g_wc_array =  [ 0.1, 0.2, 0.3,  1, 10, 100]
# g_wc_array = [10]
y_max_array = [   5,   5,   5,  5,  1, 0.4]
n_graph_array = [  20,  20,   20,  16,  21, 30]
# y_max_array = [1]

k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)

file_location = "/home/mtayl29/single_electron_polariton_methods/erf-potential-modes/"

for ijk in range(len(g_wc_array)):
    g_wc = g_wc_array[ijk]
    y_max = y_max_array[ijk]

    DIR = f"{file_location}plot_data_disp/"
    sp.call(f"mkdir -p {DIR}", shell=True)
    #######
    NPol = nf * n_kappa

    EPol = np.zeros(( len(k_points), NPol , 2))
    for k_ind, k in enumerate( k_points ):
        print (k)
        data = np.loadtxt( f"{file_location}/data/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", delimiter = ',' )
        EPol[k_ind,:,0] = data[:,0] + 0.3651387160346118
        EPol[k_ind,:,1] = data[:,1]


    # plt.style.use('dark_background')
    fs = 23
    n_states = n_graph_array[ijk]
    plot_states = np.arange( n_states )
    fig, ax = plt.subplots()

    cols = np.ndarray.flatten(((EPol[1:,:n_states,1] + EPol[:-1,:n_states,1]) / 2 ).T)

    for lmn in range(n_states):
        x = k_points
        y = EPol[:,lmn,0]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if lmn > 0:
            all_segments=np.concatenate((all_segments, segments), axis=0)
        else:
            all_segments = segments

    lc = LineCollection(all_segments, cmap='jet')
    lc.set_array(cols)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    plt.ylim(0.0,y_max)
    plt.xlim(min(k_points),max(k_points))
    cbar = fig.colorbar(line,ax=ax)
    # plt.rcParams["figure.figsize"] = (2,1.5)
    # plt.xlabel("$k$ (a.u.)",fontsize=fs)
    plt.yticks(fontsize = fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.xticks(ticks = [- np.pi / 4,0, np.pi / 4], labels = ["$-\pi/a_0$", "0", "$\pi/a_0$"],fontsize = fs)
    # plt.title(f"$g / \omega_c =$ {g_wc}",fontsize=fs)
    # plt.ylabel("Energy (a.u.)",fontsize=fs)
    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}.jpg",dpi=600)
    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}.svg",format='svg')

    # output = np.zeros(( len(k_points), NPol, 2 ))
    # # output[:,0,0] = k_points
    # output[:,:,:] = EPol
    # np.savetxt( f"{DIR}/plot_data_{g_wc}.dat", output)

    plt.figure().clear()

