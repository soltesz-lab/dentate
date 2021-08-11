#!/bin/bash

export DATA_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1000016 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=$HOME/src/model/DGC/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name GC \
        --syn-mech-name AMPA \
        --erev 0 \
        --v-init -70 \
        --syn-weight 1 \
        -m psp

mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1000016 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=$HOME/src/model/DGC/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name MC \
        --syn-mech-name AMPA \
        --erev 0 \
        --v-init -70 \
        --syn-weight 1 \
        -m psp


