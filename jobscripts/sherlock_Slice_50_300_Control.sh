#!/bin/bash
#
#SBATCH -J dentate_Slice_50_300_Control
#SBATCH -o ./results/dentate_Slice_50_300_Control.%j.o
#SBATCH -n 128
#SBATCH -t 16:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

mkdir -p ./results/Slice_50_300_Control_$SLURM_JOB_ID

module load openmpi/1.8.3/gcc

mpirun nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/sherlock_Slice_50_300_Control.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Slice_50_300_Control_$SLURM_JOB_ID\"" \
main.hoc
