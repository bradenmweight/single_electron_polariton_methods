#!/bin/bash 
#SBATCH -p action -A action 
#SBATCH -J 15_erf_potential 
#SBATCH -o output15.slurm 
#SBATCH -t 2-00:00:00 
#SBATCH -N 1 
#SBATCH --ntasks-per-node=24 
#SBATCH --mem 60GB 
 
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
 
python3 main.py 1.875 2.0 4 1024 51