#!/bin/bash
#SBATCH -p action
#SBATCH -J TD
#SBATCH -o output.slurm
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --mem 40GB

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

python3 Asym_Decoup_with_Arb_Vx.py 0.0

for j in {1..10..1}; do
    A=$( bc -l <<< $j*0.001 )
    echo "A0 = ${A}"
    python3 Asym_Decoup_with_Arb_Vx.py $A
done

for j in {1..10..1}; do
    A=$( bc -l <<< $j*0.01 )
    echo "A0 = ${A}"
    python3 Asym_Decoup_with_Arb_Vx.py $A
done

for j in {1..10..1}; do
    A=$( bc -l <<< $j*0.1 )
    echo "A0 = ${A}"
    python3 Asym_Decoup_with_Arb_Vx.py $A
done

for j in {1..10..1}; do
    A=$( bc -l <<< $j*1 )
    echo "A0 = ${A}"
    python3 Asym_Decoup_with_Arb_Vx.py $A
done

for j in {1..10..1}; do
    A=$( bc -l <<< $j*10 )
    echo "A0 = ${A}"
    python3 Asym_Decoup_with_Arb_Vx.py $A
done



