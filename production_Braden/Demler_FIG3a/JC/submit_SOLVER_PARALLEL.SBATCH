#!/bin/bash
####SBATCH -p action
####SBATCH -p exciton -A exciton
####SBATCH -p standard
#SBATCH -p debug
#SBATCH -J PF
#SBATCH -o output.slurm
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem 30GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 One-Electron_Polariton_Solver_Parallel.py



