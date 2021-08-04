#!/bin/bash

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export UCX_TLS="knem,dc_x"

export DATA_PREFIX=$SCRATCH/striped/dentate

ibrun -n 1 python3 ./cell_clamp.py \
        -g 1019046 \
        --population MC \
        --config-prefix config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=$HOME/model/dgc/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name HC \
        --syn-mech-name AMPA \
        --swc-type basal \
        --erev 0 \
        --v-init -60 \
        --syn-weight 1 \
        -m psp


