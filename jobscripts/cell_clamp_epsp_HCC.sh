#!/bin/bash


python3 ./cell_clamp.py \
        -g 1043250\
        --population HCC \
        --config-prefix $HOME/src/model/dentate/config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --dataset-prefix /media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG \
        --template-paths=$HOME/src/model/DGC/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name MPP \
        --syn-mech-name AMPA \
        --swc-type apical \
        --erev 0 \
        --v-init -60 \
        --syn-weight 1 \
        -m psp


