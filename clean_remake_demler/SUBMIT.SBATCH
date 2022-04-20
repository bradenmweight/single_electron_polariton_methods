#!/bin/bash
##SBATCH -p exciton -A exciton
#SBATCH -p standard
#SBATCH -J TD
#SBATCH -o output.slurm
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem 60GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 main.py