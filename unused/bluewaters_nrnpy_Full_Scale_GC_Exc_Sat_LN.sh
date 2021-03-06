#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=2048:ppn=16:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=9:00:00
### set the job name
#PBS -N dentate_Full_Scale_GC_Exc_Sat_LN
### set the job stdout and stderr
#PBS -e ./results/dentate.$PBS_JOBID.err
#PBS -o ./results/dentate.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A bayj


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

set -x

export SCRATCH=/projects/sciteam/bayj
export NEURONROOT=$SCRATCH/nrn
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

echo python is `which python2.7`

results_path=./results/Full_Scale_GC_Exc_Sat_LN_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

## Necessary for correct loading of Darshan with LD_PRELOAD mechanism
##export PMI_NO_FORK=1
##export PMI_NO_PREINITIALIZE=1

aprun -n 32768 -b -- bwpy-environ -- \
    python2.7 ./scripts/main.py  \
    --config-file=Full_Scale_GC_Exc_Sat_LN.yaml  \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH" \
    --results-path=$results_path \
    --io-size=256 \
    --tstop=7500 \
    --v-init=-75 \
    --results-write-time=600 \
    --stimulus-onset=50.0 \
    --max-walltime-hours=8.9 \
    --vrecord-fraction=0.001 \
    --node-rank-file=parts_GC_Exc.32768 \
    --verbose

