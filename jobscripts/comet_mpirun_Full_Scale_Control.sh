#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 24:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

results_path=./results/Full_Scale_Control_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

runhoc="./jobscripts/comet_Full_Scale_Control_run_${SLURM_JOB_ID}.hoc"

sed -e "s/JOB_ID/$SLURM_JOB_ID/g" ./jobscripts/comet_Full_Scale_Control_run.hoc > $runhoc

ibrun ./mechanisms/x86_64/special -mpi -nobanner -nogui $runhoc


