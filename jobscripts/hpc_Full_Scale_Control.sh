#!/bin/bash
#
#$ -q som,asom,pub64
#$ -pe mpi 256
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N dentate_Full_Scale_Control
#$ -o ./results/dentate_Full_Scale_Control.$JOB_ID.o
#$ -R y

module load neuron/7.4alpha

mkdir -p ./results/Full_Scale_Control_$JOB_ID

mpirun -np $CORES nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/hpc_Full_Scale_Control.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/Full_Scale_Control_$JOB_ID\"" \
main.hoc
