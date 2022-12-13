#!/bin/bash
####SBATCH -p action
####SBATCH -p exciton -A exciton
#SBATCH -p standard
####SBATCH -p debug
#SBATCH -J PF
#SBATCH -o output.slurm
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem 100GB

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=36

python3 One-Electron_Polariton_Solver_Parallel.py



