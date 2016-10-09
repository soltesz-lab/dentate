#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=356:ppn=1:xk
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=1:30:00
### set the job name
#PBS -N dentate_gpu_Full_Scale_Control
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027


module unload PrgEnv-cray 
module unload PrgEnv-intel
module load PrgEnv-pgi
module load cudatoolkit

set -x

cd $PBS_O_WORKDIR

results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path

mkdir -p $results_path

runhoc="./run.hoc"

aprun -n 356 buildgpu/bin/coreneuron_exec -mpi -d coredat \
  -o $results_path --cell_permute=1 -e 1000 --voltage=-75. --gpu


