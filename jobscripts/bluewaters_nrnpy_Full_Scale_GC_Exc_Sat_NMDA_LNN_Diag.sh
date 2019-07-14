#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=2048:ppn=16:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=10:00:00
### set the job name
#PBS -N dentate_Full_Scale_GC_Exc_Sat_NMDA_LNN_Diag
### set the job stdout and stderr
#PBS -e ./results/dentate.$PBS_JOBID.err
#PBS -o ./results/dentate.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A bayj


module load bwpy/2.0.1

set -x
export LC_ALL=en_IE.utf8
export LANG=en_IE.utf8
export SCRATCH=/projects/sciteam/bayj
export NEURONROOT=$SCRATCH/nrnintel3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

echo python is `which python2.7`

results_path=./results/Full_Scale_GC_Exc_Sat_NMDA_LNN_Diag_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

aprun -n 32768 -b -- bwpy-environ -- \
    python3.6 ./scripts/main.py  \
    --config-file=Full_Scale_GC_Exc_Sat_NMDA_LNN_Diag.yaml  \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH" \
    --results-path=$results_path \
    --io-size=256 \
    --tstop=9500 \
    --v-init=-75 \
    --results-write-time=600 \
    --stimulus-onset=50.0 \
    --max-walltime-hours=9.9 \
    --vrecord-fraction=0.001 \
    --node-rank-file=parts_GC_Exc.32768 \
    --verbose

