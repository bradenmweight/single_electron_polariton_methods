#!/bin/bash
#SBATCH -p action -A action
#SBATCH -J TD
#SBATCH -o output3.slurm
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem 60GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 main.py