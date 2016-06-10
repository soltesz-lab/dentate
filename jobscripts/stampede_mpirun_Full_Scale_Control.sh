#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH -n 512 -N 256
#SBATCH -p normal
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

mkdir -p ./results/Full_Scale_Control_$SLURM_JOB_ID

runhoc="./jobscripts/stampede_Full_Scale_Control_run_${SLURM_JOB_ID}.hoc"

sed -e "s/JOB_ID/$SLURM_JOB_ID/g" ./jobscripts/stampede_Full_Scale_Control_run.hoc > $runhoc

ibrun tacc_affinity ./mechanisms/x86_64/special -mpi $runhoc


