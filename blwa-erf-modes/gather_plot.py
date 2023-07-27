import numpy as np
import subprocess as sp
import os

nk = 1024
g_min_log = -2
g_max_log =  2
ng = 32 * 8 + 1

wc = 1
nf = 7
n_kappa = 101
BASIS = "RAD"
a_0 = 4

k_points = np.linspace(-np.pi / a_0, np.pi / a_0, nk)
# g_wc_array = 10 ** ( np.linspace((g_min_log), (g_max_log), ng , endpoint=False))
g_wc_array = 10 ** ( np.linspace((g_min_log), (g_max_log), ng))

file_location = "/scratch/mtayl29/single_electron_polariton_methods/blwa-erf-modes/"

DIR = f"{file_location}gather_data/"
sp.call(f"mkdir -p {DIR}", shell=True)

for ijk in np.flip(range(len(g_wc_array))):
    g_wc = g_wc_array[ijk]
    NPol = nf * n_kappa
    print(g_wc)

    if os.path.isfile(f"{DIR}g{'%s' % float('%.4g' % g_wc)}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy"):
        print(f"g/w already exists")
        if ijk == np.max(range(len(g_wc_array))):
            EPol = np.load( f"{DIR}g{g_wc}_nk{nk}_nf{nf}_wc{wc}_a0{a_0}_nkappa{n_kappa}.npy")
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