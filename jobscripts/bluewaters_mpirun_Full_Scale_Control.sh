#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=256:ppn=2:xe
### set the wallclock time
#PBS -l walltime=18:00:00
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

runhoc="./jobscripts/bluewaters_Full_Scale_Control_run_${PBS_JOBID}.hoc"

sed -e "s/JOB_ID/$PBS_JOBID/g" ./jobscripts/bluewaters_Full_Scale_Control_run.hoc > $runhoc

aprun -n 512 ./mechanisms/x86_64/special -mpi -nobanner -nogui $runhoc

mv ${PBS_JOBID}.err ${PBS_JOBID}.out ./results/Full_Scale_Control_$PBS_JOBID
