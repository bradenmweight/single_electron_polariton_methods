#!/bin/bash
#SBATCH -p exciton -A exciton
#SBATCH -J TD
#SBATCH -o output3.slurm
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --mem 30GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 main_disp.py
# python3 main.py