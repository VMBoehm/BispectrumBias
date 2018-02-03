#!/bin/bash -l
#SBATCH -e /global/homes/v/vboehm/N32theory/runs/error_N32
#SBATCH -o /global/homes/v/vboehm/N32theory/runs/out_N32 
#SBATCH -N 4         #Use 4 nodes
#SBATCH -t 18:00:00  #Set 4 hour limit
#SBATCH -q regular   #Submit to the regular QOS
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes
#SBATCH -A mp107 #use CMB repo
srun -n 128 -c 2 python RunTotBias.py
