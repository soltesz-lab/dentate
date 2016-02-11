#!/bin/bash
#
#$ -q som,asom,pub64
#$ -pe mpi 192
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N dentate_Slice_50_500_Epileptic
#$ -o ./results/dentate_Slice_50_500_Epileptic.$JOB_ID.o
#$ -R y

module load neuron/7.4alpha

mkdir -p ./results/Slice_50_500_Epileptic_$JOB_ID

hg manifest | tar zcf ./results/Slice_50_500_Epileptic_$JOB_ID/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf $PWD/results/Slice_50_500_Epileptic_$JOB_ID/dgc.tgz --files-from=/dev/stdin

mpirun -np $CORES nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/hpc_Slice_50_500_Epileptic.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Slice_50_500_Epileptic_$JOB_ID\"" \
main.hoc
