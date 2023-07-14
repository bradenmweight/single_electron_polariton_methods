import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scsp
import math
import subprocess as sp

a_0 = 4
v_0 = 5 / 27.2112
n_kappa = 10001
dark = True
n_cells = 3
n_kappa_graph = n_kappa * n_cells
model = 'erf'
plot_cosine = True
plot_inf = True

DIR = f"plot_{model}_pot/"
sp.call(f"mkdir -p {DIR}", shell=True)

# r_0_array = [0.1, 2, 10, 100]
r_0_array = [10]
V_x = np.zeros((n_kappa_graph, len(r_0_array)))

for r_0_ind, r_0 in enumerate(r_0_array):

    kappa_grid = 2 * np.pi / a_0 * np.linspace(-(n_kappa - 1) / 2, (n_kappa - 1) / 2, n_kappa)
    padding = np.zeros((n_cells))
    padding[0] = 1

    if model == 'erf':
        Z = np.pi / (scsp.gammaincc(1e-12,(2 * np.pi / a_0 / 2 / r_0)**2) * math.gamma(1e-12)) * v_0
        print(f"r_0 = {r_0} and Z = {Z}")
        V_k = -Z / 2 / np.pi * scsp.gammaincc(1e-12,(kappa_grid / 2 / r_0)**2) * math.gamma(1e-12)
    elif model == 'coulomb':
        q = 0.1
        V_k = - 4 * np.pi * q**2 / (kappa_grid+1e-10)**2

    V_k = np.kron(V_k,padding)
    min_ind = np.argmin(V_k)
    V_k[min_ind] = 0

    v_x = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(V_k)))
    phaseramp = np.linspace(-np.pi * (n_cells+1)/2, np.pi* (n_cells+1)/2, n_kappa_graph)
    # phaseramp = np.zeros((n_kappa_graph))
    V_x[:, r_0_ind] = np.real(v_x * np.exp(-1j * phaseramp))
    V_x[:, r_0_ind] = V_x[:, r_0_ind] - np.max(V_x[:, r_0_ind])
    x = np.linspace(-a_0/2 * n_cells, a_0/2* n_cells, n_kappa_graph)
    cos_graph = -v_0 * np.cos(x * (2 * np.pi / a_0)) - v_0

black = '0'
fsize = (8, 4)
label = 'bright'
if dark:
    plt.style.use('dark_background')
    black = '0.95'
    fsize = (15,4)
    label = 'dark'

plt.figure(figsize=fsize)
if plot_cosine:
    plt.plot(x, cos_graph, ':', markersize=3, linewidth=6, label=r'$r_0 \ll 1$', color = black)
# colors_line = ['0.0','0.3','0.5','0.75']
if plot_inf:
    inf_graph = np.zeros((n_kappa))
    inf_graph[n_kappa//2] = -1e10
    inf_graph = np.kron(np.ones((n_cells)), inf_graph)
    plt.plot(x, inf_graph, linewidth=3, label=r'$r_0 \;\rightarrow \infty$', color = black)
for r_0_ind in range(len(r_0_array)):
    plt.plot(x, V_x[:, r_0_ind], linewidth=3, label=f'$r_0$ = {r_0_array[r_0_ind]}', color= 'red')#, color = colors_line[r_0_ind])
    plt.ylim(-1, 0)

plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05,
                    bottom=0.14,
                    right=0.95,
                    top=0.92,
                    wspace=0.2,
                    hspace=0.2)
plt.xlabel('x (a.u.)',fontsize=15)
plt.ylabel('V(x)',fontsize=15)
plt.savefig(f"{DIR}/{model}_{label}.jpg",dpi=600)
plt.savefig(f"{DIR}/{model}_{label}.svg",format='svg')