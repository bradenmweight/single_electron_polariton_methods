#!/bin/bash
#SBATCH -p action -A action
#SBATCH -J gather_data
#SBATCH -o gather.out
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 60GB

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 gather_plot.py
# python3 plot_video_frames.py
# python3 make_gif.py