#!/bin/bash
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export DATA_PREFIX=/scratch1/03320/iraikov/striped/dentate 

ibrun -n 1 python3 ./cell_clamp.py \
        -g 1043250\
        --population HCC \
        --config-prefix $HOME/model/dentate/config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=$HOME/model/DGC/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name MPP \
        --syn-mech-name AMPA \
        --swc-type apical \
        --erev 0 \
        --v-init -60 \
        --syn-weight 1 \
        --syn-weight 20 \
        -m psp


