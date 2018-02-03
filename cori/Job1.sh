#!/bin/bash -l
#SBATCH -e /u/vboehm/error_con1
#SBATCH -o /u/vboehm/out_con1
#SBATCH -D /u/vboehm/CosmoCodes/
#SBATCH --time=23:00:00
#SBATCH --job-name=conv1
#SBATCH --nodes=8
#SBATCH --ntasks=256
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boehmvanessa@gmail.com

module purge
module load anaconda
module load impi
unset I_MPI_HYDRA_BOOTSTRAP
unset I_MPI_PMI_LIBRARY

srun -n 256 python RunTypeA1.py >TypeAtest.log
