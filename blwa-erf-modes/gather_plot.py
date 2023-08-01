import numpy as np
import subprocess as sp
import os
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

nk = 512
g_min_log =  0
g_max_log =  2

plot_video_frames = True
dark = False
g_skip = 58

ng = 4
n_sections = 16
end = False
ng_total = ng * n_sections + int(end)

wc = 1
nf = 3
n_kappa = 501
BASIS = "RAD"
a_0 = 4

k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)
g_wc_array = 10 ** ( np.linspace((g_min_log), (g_max_log), ng_total, endpoint=end))
y_max_array = np.ones((ng_total)) * 5.0
n_graph_array = (np.ones((ng_total)) * 200).astype(int)

file_location = "/scratch/mtayl29/single_electron_polariton_methods/blwa-erf-modes/"

DIR = f"{file_location}gather_data/"
sp.call(f"mkdir -p {DIR}", shell=True)

def plot_for_coupling(g_wc):
    EPol = np.load( f"{DIR}g{'%s' % float('%.4g' % g_wc)}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy")
    # print(f"g/w = {g_wc}")

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

    for lmn in range(all_segments.shape[0]):
        if all_segments[lmn,0,1] > 5.05:
            cols[lmn] = np.min(cols)

    lc = LineCollection(all_segments, cmap='jet')
    lc.set_array(cols)
    # lc.set_linewidth(3)
    return lc

for ijk in np.flip(range(len(g_wc_array))):
    g_wc = g_wc_array[ijk]
    NPol = nf * n_kappa
    print(g_wc)

    if os.path.isfile(f"{DIR}g{'%s' % float('%.4g' % g_wc)}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy"):
        print(f"g/w already exists")
        if ijk == np.max(range(len(g_wc_array))):
            EPol = np.load( f"{DIR}g{'%s' % float('%.4g' % g_wc)}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy")
            e_min = np.min(EPol[:,0,0])
    else:
        EPol = np.zeros(( len(k_points), NPol , 2))
        for k_ind, k in enumerate( k_points ):
            # print (k)
            data = np.loadtxt( f"{file_location}data/E_{BASIS}_k{np.round(k,3)}_{nf}_{n_kappa}_gwc{np.round(g_wc,7)}_wc{np.round(wc,4)}.dat", delimiter = ',' )
            EPol[k_ind,:,0] = data[:,0]
            EPol[k_ind,:,1] = data[:,2]
        if ijk == np.max(range(len(g_wc_array))):
            e_min = np.min(EPol[:,0,0])
        EPol[:,:,0] = EPol[:,:,0] - e_min
    
        np.save( f"{DIR}g{'%s' % float('%.4g' % g_wc)}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy", EPol)

if plot_video_frames:

    IMG_DIR = f"{file_location}video_frames_{n_kappa}/"
    sp.call(f"mkdir -p {IMG_DIR}", shell=True)

    label = 'bright'
    black = 'k'
    if dark:
        plt.style.use('dark_background')
        label = 'dark'
        black = '0.95'

    k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)

    for ijk in (range(len(g_wc_array))):
        if ijk%g_skip == 0:
            g_wc = g_wc_array[ijk]
            print(f"Plotting g_wc = {g_wc}")
            y_max = y_max_array[ijk]
            sp.call(f"mkdir -p {DIR}", shell=True)
            #######

            fs = 23
            n_states = n_graph_array[ijk]
            plot_states = np.arange( n_states )
            fig, ax = plt.subplots()

            lc = plot_for_coupling(g_wc)

            line = ax.add_collection(lc)
            cbar = fig.colorbar(line,ax=ax)
            
            # cbar.ax.tick_params(labelsize=fs)

            ax.set_ylim(ymin=-2.0,ymax=y_max)
            # plt.yscale('log')
            plt.xlim(min(k_points),max(k_points))
            # plt.rcParams["figure.figsize"] = (2,1.5)
            plt.xlabel("$k$ (a.u.)",fontsize=fs)
            # plt.yticks(fontsize = fs, ticks = np.linspace(0 , y_max, nticks[ijk], True))
            plt.yticks(fontsize = fs)
            plt.xticks(ticks = [- np.pi / a_0, 0, np.pi / a_0], labels = ["$-\pi/a_0$", "0", "$\pi/a_0$"],fontsize = fs)
            plt.title(f"$g / \omega_c =$ {'%s' % float('%.3g' % g_wc)}",fontsize=fs)
            plt.ylabel("Energy (a.u.)",fontsize=fs)
            plt.subplots_adjust(left=0.17,
                            bottom=0.18,
                            right=0.93,
                            top=0.87,
                            wspace=0.2,
                            hspace=0.2)

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

            plt.savefig(f"{IMG_DIR}/{ijk:006}.png",dpi=600)
            # plt.savefig(f"{IMG_DIR}/disp_plot_g{np.round(g_wc,3)}_nf{nf}_{label}.svg",format='svg')

            # np.save( f"{DIR}plot_data_{g_wc}_{nk}_{a_0}_{n_kappa}_{nf}_{wc}.npy", EPol)

            plt.figure().clear()

        # sp.call(f"convert -delay 10 {IMG_DIR}*.png dispersion.gif")   