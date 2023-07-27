#!/bin/bash
#SBATCH -p preempt
#SBATCH -J gather_data
#SBATCH -o gather.out
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 60GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 plot_video_frames.py