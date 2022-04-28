#!/software/anaconda3/2020.07/bin/python
#SBATCH -A exciton -p exciton 
#SBATCH -o output.log
#SBATCH --mem-per-cpu=12GB
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -N 1
import numpy as np
import time , sys, os
sys.path.append(os.popen("pwd").read().replace("\n",""))
t0 = time.time()
print ()
#----------------------------  SBATCH  ---------------------------------------------------
sbatch = [i for i in open('dw.py',"r").readlines() if i[:10].find("#SBATCH") != -1 ]
cpu   = int(sbatch[-2].split()[-1].replace("\n","")) #int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
nodes = int(sbatch[-1].split()[-1].replace("\n",""))

print (os.environ['SLURM_JOB_NODELIST'], os.environ['SLURM_JOB_CPUS_PER_NODE'])
print (f"nodes : {nodes} | cpu : {cpu}")
procs = cpu * nodes

import numpy as np  
import math 
from numpy import linalg as LA
import sys
#----------------------------------------
# Matrix Diagonalization

def Diag(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#----------------------------------------

# Kinetic energy for electron 
def Te(re):
 N = float(len(re))
 mass = 1.0
 Tij = np.zeros((int(N),int(N)))
 Rmin = float(re[0])
 Rmax = float(re[-1])
 step = float((Rmax-Rmin)/N)
 K = np.pi/step

 for ri in range(int(N)):
  for rj in range(int(N)):
    if ri == rj:  
     Tij[ri,ri] = (0.5/mass)*K**2.0/3.0*(1+(2.0/N**2)) 
    else:    
     Tij[ri,rj] = (0.5/mass)*(2*K**2.0/(N**2.0))*((-1)**(rj-ri)/(np.sin(np.pi*(rj-ri)/N)**2)) 
 return Tij
#---------------------------------

def Ve(re):
    Vij =  np.zeros((len(re),len(re)))
    beta = 3 # 50
    gamma = 3.85 # 95 
    omega = 1
    for ri in range(len(re)):
        Vij[ri,ri] = - beta * re[ri]**2 / 2 + gamma * re[ri]**4 /4
        # Vij[ri,ri] = 0.5 * omega**2 * re[ri]**2
    return Vij

def Hel(param):
    re = param.re 
    V = Ve(re) 
    T = Te(re) 
    He = T + V 
    E, V = Diag(He)
    return mu(V, param), E[:param.states]
 
def mu(V, param):
    dm = np.zeros((param.states,param.states)) 
    for i in range(param.states):
        for j in range(param.states):
            dm[i,j] = - np.sum(param.re  * V[:,i]  * V[:,j] ) 
    return dm 

au = 0.529177249 # A to a.u.
# Parameters 2b
class param:
    states = 20
    mass = 1836.0
    points = 256
    # re = np.linspace(- 10, 10, points)
    re = np.linspace(- 2, 2, points)


if __name__=="__main__":
    points = param.points
    fob = open(f"pes_{points}_shallow.txt", "w+") 
    dmfob = open(f"dm_{points}_shallow.txt", "w+") 
    dm, Ei = Hel(param) 
    fob.write("\t" + "\t".join(Ei.astype("str")) + "\n")
    dmfob.write("\t" + "\t".join(dm.flatten().astype("str")) + "\n")
    fob.close()
    dmfob.close()

    t2 = time.time() - t0 
    print (f"Time taken: {t2} s")