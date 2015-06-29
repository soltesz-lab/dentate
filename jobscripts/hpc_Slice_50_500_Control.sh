#!/bin/bash
#
#$ -q som,asom,pub64
#$ -pe mpi 320
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N dentate_Slice_50_500_Control
#$ -o ./results/dentate_Slice_50_500_Control.$JOB_ID.o
#$ -R y

module load neuron/7.4alpha

mkdir -p ./results/Slice_50_500_Control_$JOB_ID

hg manifest | tar zcf ./results/Slice_50_500_Control_$JOB_ID/dentate.tgz --files-from=/dev/stdin

mpirun -np $CORES nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/hpc_Slice_50_500_Control.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Slice_50_500_Control_$JOB_ID\"" \
main.hoc
