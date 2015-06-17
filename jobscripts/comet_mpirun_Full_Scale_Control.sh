#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

mkdir -p ./results/Full_Scale_Control_$SLURM_JOB_ID

ibrun ./mechanisms/x86_64/special -mpi -nobanner -nogui -c "strdef parameters" -c "parameters=\"./parameters/Full_Scale_Control.hoc\"" -c "strdef resultsPath" -c "resultsPath=\"./results/Full_Scale_Control_$SLURM_JOB_ID\"" main.hoc
