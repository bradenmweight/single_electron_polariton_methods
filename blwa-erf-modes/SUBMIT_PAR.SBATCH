#!/bin/bash 
#SBATCH -p action -A action 
#SBATCH -J 7_erf_potential 
#SBATCH -o output7.slurm 
#SBATCH -t 2-00:00:00 
#SBATCH -N 1 
#SBATCH --ntasks-per-node=24 
#SBATCH --mem 60GB 
 
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
 
python3 main.py 1.5 2.0 32 1024