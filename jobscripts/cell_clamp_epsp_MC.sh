#!/bin/bash

export DATA_PREFIX=datasets
# 4781 GC synapses
mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1014509 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config Network_Clamp_Slice_neg10_pos10um_SynExp3NMDA2fd_CLS_IN_PR.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=templates \
        --results-path=results/cell_clamp \
        --presyn-name GC \
        --syn-mech-name AMPA \
        --erev 0 \
        --v-init -70 \
        --syn-count 900 \
        --syn-weight 1 \
        -m psp
# 497 CA3c synapses
mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1014509 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config Network_Clamp_Slice_neg10_pos10um_SynExp3NMDA2fd_CLS_IN_PR.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=templates \
        --results-path=results/cell_clamp \
        --presyn-name CA3c \
        --syn-mech-name AMPA \
        --erev 0 \
        --v-init -70 \
        --syn-count 3 \
        --syn-weight 1 \
        -m psp

# 1536 MC synapses
mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1014509 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config Network_Clamp_Slice_neg10_pos10um_SynExp3NMDA2fd_CLS_IN_PR.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=templates \
        --results-path=results/cell_clamp \
        --presyn-name MC \
        --syn-mech-name AMPA \
        --erev 0 \
        --v-init -70 \
        --syn-weight 1 \
        --syn-count 3 \
        -m psp


