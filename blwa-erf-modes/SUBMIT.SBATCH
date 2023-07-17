#!/bin/bash
#SBATCH -p action -A action
#SBATCH -J erf_potential
#SBATCH -o output.slurm
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem 60GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 main.py 10.0 100.0 32 1024