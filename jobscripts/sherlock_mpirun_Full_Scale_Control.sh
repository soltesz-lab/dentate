#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH -n 128
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

set -x

mkdir -p ./results/Full_Scale_Control_$SLURM_JOB_ID

module load openmpi/1.7.4/gcc

mpirun nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/Full_Scale_Control.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Full_Scale_Control_$SLURM_JOB_ID\"" \
main.hoc
