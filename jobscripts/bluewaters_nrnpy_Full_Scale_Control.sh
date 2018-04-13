#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=188:ppn=16:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=2:30:00
### set the job name
#PBS -N dentate_Full_Scale_Control
### set the job stdout and stderr
#PBS -e ./results/dentate.$PBS_JOBID.err
#PBS -o ./results/dentate.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

set -x

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

echo python is `which python`
results_path=./results/Full_Scale_Control_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

## Necessary for correct loading of Darshan with LD_PRELOAD mechanism
##export PMI_NO_FORK=1
##export PMI_NO_PREINITIALIZE=1

aprun -n 3008 -b -- bwpy-environ -- \
    python2.7 main.py  \
    --config-file=config/Full_Scale_Control.yaml  \
    --template-paths=../dgc/Mateos-Aparicio2014 \
    --dataset-prefix="$SCRATCH" \
    --results-path=$results_path \
    --io-size=256 \
    --tstop=10000 \
    --v-init=-75 \
    --stimulus-onset=150.0 \
    --max-walltime-hours=2.4 \
    --vrecord-fraction=0.001 \
    --verbose

