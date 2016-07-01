#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=256:ppn=2:xe
### set the wallclock time
#PBS -l walltime=4:00:00
### set the job name
#PBS -N dentate_Full_Scale_Control
### set the job stdout and stderr
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
### save Darshan profile data
#PBS -lgres=darshan


module swap PrgEnv-cray PrgEnv-intel

set -x

cd $PBS_O_WORKDIR

mkdir -p ./results/Full_Scale_Control_$PBS_JOBID

aprun -n 512 ./mechanisms/x86_64/special -mpi -nobanner -nogui -c "strdef parameters" -c "parameters=\"./parameters/bluewaters_Full_Scale_Control.hoc\"" -c "strdef resultsPath" -c "resultsPath=\"./results/Full_Scale_Control_$PBS_JOBID\"" main.hoc

mv ${PBS_JOBID}.err ${PBS_JOBID}.out ./results/Full_Scale_Control_$PBS_JOBID
