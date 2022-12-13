#!/bin/bash
#SBATCH -p action
####SBATCH -p exciton -A exciton
####SBATCH -p standard
####SBATCH -p debug
#SBATCH -J AD
#SBATCH -o output.slurm
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem 60GB

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=1

python3 One-Electron_Polariton_Solver_Parallel.py



