import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scsp
import math
import subprocess as sp

a_0 = 4
v_0 = 5 / 27.2112
n_kappa = 20001
n_kappa_graph = n_kappa

DIR = "plot_erf_pot/"
sp.call(f"mkdir -p {DIR}", shell=True)

r_0_array = [0.1, 2, 10, 100]
V_x = np.zeros((n_kappa, len(r_0_array)))

for r_0_ind, r_0 in enumerate(r_0_array):
    Z = np.pi / (scsp.gammaincc(1e-12,(2 * np.pi / a_0 / 2 / r_0)**2) * math.gamma(1e-12)) * v_0

    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa - 1) / 2, (n_kappa - 1) / 2, n_kappa_graph)

    V_k = -Z / 2 / np.pi * scsp.gammaincc(1e-12,(kappa_grid / 2 / r_0)**2) * math.gamma(1e-12)
    min_ind = np.argmin(V_k)
    V_k[min_ind] = 0

    v_x = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(V_k)))
    phaseramp = np.linspace(-np.pi, np.pi, n_kappa_graph)
    V_x[:, r_0_ind] = np.real(v_x * np.exp(-1j * phaseramp))
    V_x[:, r_0_ind] = V_x[:, r_0_ind] - np.max(V_x[:, r_0_ind])
    x = np.linspace(-a_0/2, a_0/2, n_kappa_graph)
    cos_graph = -v_0 * np.cos(x * (2 * np.pi / a_0)) - v_0

plt.figure(figsize=(8, 4))
plt.plot(x, cos_graph, ':', markersize=3, linewidth=6, label='cosine', color = '0.0')
colors_line = ['0.0','0.3','0.5','0.75']
for r_0_ind in range(len(r_0_array)):
    plt.plot(x, V_x[:, r_0_ind], linewidth=3, label=f'$r_0$ = {r_0_array[r_0_ind]}', color = colors_line[r_0_ind])
    plt.ylim(-1, 0)
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.125,
                    bottom=0.14,
                    right=0.9,
                    top=0.92,
                    wspace=0.2,
                    hspace=0.2)
plt.xlabel('r (a.u.)',fontsize=15)
plt.ylabel('v(r)',fontsize=15)
# plt.savefig(f"{DIR}/erf.jpg",dpi=600)
plt.savefig(f"{DIR}/erf.svg",format='svg')