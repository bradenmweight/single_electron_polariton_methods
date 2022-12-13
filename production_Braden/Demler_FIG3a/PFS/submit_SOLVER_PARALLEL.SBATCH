#!/bin/bash
####SBATCH -p action
####SBATCH -p exciton -A exciton
#SBATCH -p standard
####SBATCH -p debug
#SBATCH -J PFS
#SBATCH -o output.slurm
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=36
#SBATCH --mem 100GB

export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=1

python3 One-Electron_Polariton_Solver_Parallel.py 20 20

#for NPH in {200..250..50}; do
#    for NM in {25..200..25}; do
#
#        python3 One-Electron_Polariton_Solver_Parallel.py ${NPH} ${NM}
#        python3 plot_polaritons.py ${NPH} ${NM}
#
#    done
#done