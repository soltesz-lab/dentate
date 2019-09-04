#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=16:ppn=32:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=2:00:00
### set the job name
#PBS -N dentate_GC_Exc_Sat_DD_SLN_slice
### set the job stdout and stderr
#PBS -e ./results/dentate_slice.$PBS_JOBID.err
#PBS -o ./results/dentate_slice.$PBS_JOBID.out
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

results_path=./results/GC_Exc_Sat_DD_SLN_slice_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

aprun -n 256 -N 16 -d 2 -b -- bwpy-environ -- \
    python3.6 ./scripts/main.py \
    --arena-id=A --trajectory-id=Diag \
    --config-file=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH" \
    --results-path=$results_path \
    --io-size=24 \
    --tstop=10 \
    --v-init=-75 \
    --max-walltime-hours=1.75 \
    --cell-selection-path=./datasets/DG_slice_20190729.yaml \
    --write-selection \
    --spike-input-path="$SCRATCH/Full_Scale_Control/DG_input_spike_trains_20190724_compressed.h5" \
    --spike-input-namespace='Input Spikes' \
    --verbose

