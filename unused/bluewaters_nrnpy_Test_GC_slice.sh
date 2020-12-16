#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=100:ppn=32:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=8:00:00
### set the job name
#PBS -N dentate_Test_GC_slice
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

results_path=./results/Test_GC_slice_$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

aprun -n 800 -N 8 -d 4 -b -- bwpy-environ -- \
    python3.6 ./scripts/main.py \
    --arena-id=A --trajectory-id=Diag \
    --config-file=Network_Clamp_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH" \
    --results-path=$results_path \
    --io-size=24 \
    --tstop=10000 \
    --v-init=-75 \
    --max-walltime-hours=7.9 \
    --cell-selection-path=./datasets/DG_slice_20190917.yaml \
    --spike-input-path="$SCRATCH/Full_Scale_Control/DG_input_spike_trains_20190912_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --spike-input-attr='Spike Train' \
    --verbose

