#!/bin/bash
#
#SBATCH -J dentate_Slice_50_500_Epileptic
#SBATCH -o ./results/dentate_Slice_50_500_Epileptic.%j.o
#SBATCH -n 128
#SBATCH -t 16:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

mkdir -p ./results/Slice_50_500_Epileptic_$SLURM_JOB_ID

module load openmpi/1.8.3/gcc

hg manifest | tar zcf ./results/Slice_50_500_Epileptic_$SLURM_JOB_ID/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf $PWD/results/Slice_50_500_Epileptic_$SLURM_JOB_ID/dgc.tgz --files-from=/dev/stdin

mpirun nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/sherlock_Slice_50_500_Epileptic.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Slice_50_500_Epileptic_$SLURM_JOB_ID\"" \
main.hoc
