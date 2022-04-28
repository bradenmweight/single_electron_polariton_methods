#!/software/anaconda3/2020.07/bin/python
#SBATCH -A action -p action 
#SBATCH -o output11.log
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 3:00:00
#SBATCH -n 12
#SBATCH -N 1
import numpy as np
from multiprocessing import Pool
import time , sys, os
sys.path.append(os.popen("pwd").read().replace("\n",""))
from polariton import Ĥ, param, ĉ
from numpy import kron as ꕕ
#-------------------------------------
try: 
    nf = int(sys.argv[1]) #param.nf
    ns = int(sys.argv[2]) #param.ns
    print (f"Using matter-states: {ns}, and Fock-states: {nf}")
except:
    nf = param.nf
    ns = param.ns
    print (f"Default matter-states: {ns}, and Fock-states: {nf}")
η  = param.η_1
#-------------------------------------
t0 = time.time()
#----------------------------  SBATCH  ---------------------------------------------------
sbatch = [i for i in open('par_coup.py',"r").readlines() if i[:10].find("#SBATCH") != -1 ]
cpu   = int(sbatch[-2].split()[-1].replace("\n","")) #int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
nodes = int(sbatch[-1].split()[-1].replace("\n",""))
print (os.environ['SLURM_JOB_NODELIST'], os.environ['SLURM_JOB_CPUS_PER_NODE'])
print (f"nodes : {nodes} | cpu : {cpu}")
procs = cpu * nodes
#-------------------------------------
points = 20000

E = np.loadtxt(f"../Hm/pes_{points}.txt")
N = len(E[:])

µ = np.loadtxt(f"../Hm/dm_{points}.txt")[:]

eta = np.linspace(0.0, 0.5, 64)
#-----------------------------------------
print (f"Total states available : {N}")
#-----------------------------------------
with Pool(len(eta)) as p:
    #------ Arguments for each CPU--------
    args = []
    for j in range(len(eta)):
        par = param() 
        par.ωc = E[1] - E[0]
        par.ns = ns
        par.η_1 = eta[j]
        par.η_3 = eta[j] / np.sqrt(3)
        par.χ_1 = par.ωc * par.η_1
        par.χ_3 = par.ωc * par.η_3 * 3

        #----------------------
        H   = np.zeros((N, N))
        H[np.diag_indices(N)] = E[:]
        par.H = H[:ns,:ns]
        par.µ = µ[:].reshape((N,N))[:ns,:ns]
        param.nf = nf
        param.ns = ns
        Hij = Ĥ(par)
        args.append(Hij)

    #-------- parallelization --------------- 
    result  = p.map(np.linalg.eigh, args)
    #----------------------------------------
t2 = time.time() - t0 
print (f"Time taken: {t2} s")
print (f"Time for each point: {t2/len(eta)} s")
#------- Gather -----------------------------
temp, _ = result[0]
e_length = len(temp)
E_new = np.zeros((e_length+1, len(eta)))
U_small = np.zeros((e_length, e_length))
dse_cont = np.zeros((e_length+1, len(eta)))
coup_cont = np.zeros((e_length+1, len(eta)))

par = param() 
par.ωc = E[1] - E[0]
par.ns = ns

Iₚ = np.identity(par.nf)
par.µ = µ[:].reshape((N,N))[:ns,:ns]
â = ĉ(nf)

for etai in range(len(eta)):
    par.η_1 = eta[etai]
    par.η_3 = eta[etai] / np.sqrt(3)
    par.χ_1 = par.ωc * par.η_1
    par.χ_3 = par.ωc * par.η_3 * 3

    E_new[1:,etai] = result[etai][0]
    U_small[:,:] = result[etai][1]

    dse_cont[0,etai] = eta[etai]
    coup_cont[0,etai] = eta[etai]

    Dse3 = ꕕ(ꕕ(par.µ @ par.µ, Iₚ),Iₚ) * (par.χ_3**2/(3.0 * par.ωc))
    Coup3 = ꕕ(ꕕ(par.µ, Iₚ), (â.T + â)) * par.χ_3

    U = ꕕ(U_small[:,:],Iₚ)

    dse_tot = U.T @ Dse3 @ U 
    coup_tot = U.T @ Coup3 @ U

    # First Order
    for j in range(e_length):
        E_new[j+1, etai] += dse_tot[j,j] + coup_tot[j,j]
        dse_cont[j+1,etai] += dse_tot[j,j]
    
    # Second Order
    for j in range (e_length):
        for m in range (44):
            if  etai != 0:
                if m != j:
                    # E_new[j+1, etai] += (dse_tot[j,m] + coup_tot[j,m])*(dse_tot[m,j] + coup_tot[m,j]) / (result[etai][0][j] - result[etai][0][m])
                    E_new[j+1, etai] += (coup_tot[j,m])*(coup_tot[j,m]) / (result[etai][0][j] - result[etai][0][m])
                    dse_cont[j+1,etai] += (dse_tot[j,m])*(dse_tot[m,j]) / (result[etai][0][j] - result[etai][0][m])
                    coup_cont[j+1, etai] += (coup_tot[j,m])*(coup_tot[j,m]) / (result[etai][0][j] - result[etai][0][m])

    

        



#---- Shift ZPE ----------
E0 = (E_new[1,0]) # ZPE
E_new[1:,:] -= E0

# for n in range(len(E[1,:])):
#     E[1:,n] = E[1:,n] - E[1,n]
#-------------------------
E_new[0,:] = eta
#--------------------------------------------
np.savetxt(f"E-f{nf}-n{ns}-p{points}_abs_perturb_jm.txt", E_new.T)
np.savetxt(f"DSE-f{nf}-n{ns}-p{points}_p.txt", dse_cont.T)
np.savetxt(f"Coup-f{nf}-n{ns}-p{points}_p.txt", coup_cont.T)