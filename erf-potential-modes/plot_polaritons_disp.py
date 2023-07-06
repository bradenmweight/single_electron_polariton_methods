import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import subprocess as sp
nk = 1024
wc = 0.2
nf = 5
n_kappa = 101
BASIS = "RAD"
a_0 = 4
g_wc_array =  [ 0.1, 0.2, 0.3,  1, 10, 100]
# g_wc_array = [0.1]
y_max_array = [   2,   2,   2,  2,  1, 0.4]
nticks = [6,6,6,6,6,5]
n_graph_array = [  20,  20,   20,  16,  21, 30]
# y_max_array = [0.4]
dark = True
color = True

label = 'bright'
black = 'k'
if dark:
    plt.style.use('dark_background')
    label = 'dark'
    black = '0.95'

k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)

file_location = "/home/mtayl29/single_electron_polariton_methods/erf-potential-modes/"

e_min = 0
for ijk in np.flip(range(len(g_wc_array))):
    g_wc = g_wc_array[ijk]
    y_max = y_max_array[ijk]

    DIR = f"{file_location}plot_data_disp/"
    sp.call(f"mkdir -p {DIR}", shell=True)
    #######
    NPol = nf * n_kappa

    try:
        EPol = np.load( f"{DIR}plot_data_{g_wc}_{nk}_{a_0}_{n_kappa}_{nf}_{wc}.npy")
        print(f"g/w = {g_wc}")
    except:
        EPol = np.zeros(( len(k_points), NPol , 2))
        for k_ind, k in enumerate( k_points ):
            print (k)
            data = np.loadtxt( f"{file_location}data/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", delimiter = ',' )
            EPol[k_ind,:,0] = data[:,0]
            EPol[k_ind,:,1] = data[:,1]
        if ijk == np.max(range(len(g_wc_array))):
            e_min = np.min(EPol[:,0,0])
        EPol[:,:,0] = EPol[:,:,0] - e_min


    # plt.style.use('dark_background')
    fs = 23
    n_states = n_graph_array[ijk]
    plot_states = np.arange( n_states )
    fig, ax = plt.subplots()

    if color:
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

        lc = LineCollection(all_segments, color = '0.95') #cmap='jet')
        lc.set_array(cols)
        # lc.set_linewidth(3)
        line = ax.add_collection(lc)
        # cbar = fig.colorbar(line,ax=ax)
        # cbar.ax.tick_params(labelsize=fs)
    else:
        for state in plot_states:
            plt.plot( k_points, EPol[:,state], color=black)

    plt.ylim(0.0,y_max)
    plt.xlim(min(k_points),max(k_points))
    # plt.rcParams["figure.figsize"] = (2,1.5)
    # plt.xlabel("$k$ (a.u.)",fontsize=fs)
    plt.yticks(fontsize = fs, ticks = np.linspace(0 , y_max, nticks[ijk], True))
    plt.xticks(ticks = [- np.pi / 4,0, np.pi / 4], labels = ["$-\pi/a_0$", "0", "$\pi/a_0$"],fontsize = fs)
    # plt.title(f"$g / \omega_c =$ {g_wc}",fontsize=fs)
    # plt.ylabel("Energy (a.u.)",fontsize=fs)

    if dark:
        # plt.title(f"$g_0 / \omega_0 =$ {g_wc}",fontsize=fs, pad = 15)
        plt.ylabel("Energy (a.u.)",fontsize=fs)
        plt.xlabel("$k$",fontsize=fs)
        # cbar.set_label("$<a^+a>$", fontsize = fs)
        plt.subplots_adjust(left=0.17,
                    bottom=0.18,
                    right=0.93,
                    top=0.87,
                    wspace=0.2,
                    hspace=0.2)

    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}_{label}.jpg",dpi=600)
    plt.savefig(f"{DIR}/disp_plot_g{np.round(g_wc,3)}_{label}.svg",format='svg')

    np.save( f"{DIR}plot_data_{g_wc}_{nk}_{a_0}_{n_kappa}_{nf}_{wc}.npy", EPol)

    plt.figure().clear()

