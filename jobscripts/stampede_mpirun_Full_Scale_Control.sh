#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH -n 512
#SBATCH -p normal
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

mkdir -p ./results/Full_Scale_Control_$SLURM_JOB_ID

ibrun tacc_affinity ./mechanisms/x86_64/special -mpi -nobanner -nogui \
 -c "strdef parameters" \ -c "parameters=\"./parameters/stampede_Full_Scale_Control.hoc\"" \
 -c "strdef resultsPath" -c "resultsPath=\"./results/Full_Scale_Control_$SLURM_JOB_ID\"" main.hoc


