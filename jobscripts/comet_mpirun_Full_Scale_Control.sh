#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 14:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

results_path=./results/Full_Scale_Control_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun ./mechanisms/x86_64/special -mpi -nobanner -nogui -c "strdef parameters" -c "parameters=\"./parameters/comet_Full_Scale_Control.hoc\"" -c "strdef resultsPath" -c "resultsPath=\"${results_path}\"" main.hoc

