#!/bin/bash
#
#$ -q som,asom,pub64
#$ -pe mpi 256
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N dentate_slice0000
#$ -o ./results/dentate_slice0000.$JOB_ID.o
#$ -R y

#module load openmpi-1.6.5/gcc-4.7.3
#PATH=$HOME/bin/nrn/x86_64/bin:$PATH
#export PATH

module load neuron/7.4alpha

mkdir -p ./results/$JOB_ID

mpirun -np $CORES nrniv -mpi -nobanner -nogui \
-c "strdef parameters" -c "parameters=\"./parameters/slice0000.hoc\"" \
-c "strdef resultsPath" -c "resultsPath=\"./results/slice0000_$JOB_ID\"" \
main.hoc
