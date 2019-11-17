#!/bin/bash
#PBS -l nodes=1024:ppn=32:xe
#PBS -q high
#PBS -l walltime=8:00:00
#PBS -e ./results/generate_structured_weights_DG_GC.$PBS_JOBID.err
#PBS -o ./results/generate_structured_weights_DG_GC.$PBS_JOBID.out
#PBS -N generate_DG_GC_structured_weights
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

set -x
cd $PBS_O_WORKDIR


aprun -n 16384 -N 16 -d 2 -b -- bwpy-environ -- \
    python3.6 $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s LPP -s MPP \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --structured-weights-namespace='Structured Weights' \
    --output-weights-path=$SCRATCH/Full_Scale_Control/DG_GC_syn_weights_SLN_20191030.h5 \
    --weights-path=$SCRATCH/Full_Scale_Control/DG_Cells_Full_Scale_20190917.h5 \
    --connections-path=$SCRATCH/Full_Scale_Control/DG_Connections_Full_Scale_20190917.h5 \
    --input-features-path="$SCRATCH/Full_Scale_Control/DG_input_features_20190909_compressed.h5" \
    --arena-id=A \
    --io-size=256 --cache-size=10  --value-chunk-size=100000 --chunk-size=20000 --write-size=10 -v

