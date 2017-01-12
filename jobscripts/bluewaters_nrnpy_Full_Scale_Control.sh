#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=2048:ppn=1:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=3:00:00
### set the job name
#PBS -N dentate_Full_Scale_Control
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module unload gcc
module load gcc/5.1.0
module load bwpy bwpy-mpi
module load cray-hdf5-parallel

set -x

export PYTHONPATH=/projects/sciteam/baef/nrn/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=/sw/bw/bwpy/0.3.0/python-mpi/usr/lib:/sw/bw/bwpy/0.3.0/python-single/usr/lib:$LD_LIBRARY_PATH
export PATH=/projects/sciteam/baef/nrn/x86_64/bin:$PATH
export DARSHAN_LOGPATH=$PWD/darshan-logs

echo python is `which python`

results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

aprun -n 2048 nrniv -mpi -python main.py \
    --model-name=dentatenet \
    --dataset-name=Full_Scale_Control \
    --dataset-prefix="/u/sciteam/raikov/model/dentate/datasets/B2048" \
    --results-path=$results_path \
    --io-size=128 \
    --tstop=1000 \
    --v-init=-75 \
    --max-walltime-hours=3





