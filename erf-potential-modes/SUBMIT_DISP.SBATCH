#!/bin/bash
#SBATCH -p preempt
#SBATCH -J erf_potential
#SBATCH -o output_disp.slurm
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --mem 300GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 main_disp.py