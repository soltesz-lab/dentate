#!/bin/bash
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export DATA_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

mpirun -n 1 python3 ./cell_clamp.py -v \
        -g 1015624 \
        --population MC \
        --config-prefix $HOME/src/model/dentate/config \
        --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
        --dataset-prefix $DATA_PREFIX \
        --template-paths=$HOME/src/model/DGC/Mateos-Aparicio2014:templates \
        --results-path=results/cell_clamp \
        --presyn-name CA3c \
        --syn-mech-name AMPA \
        --swc-type apical \
        --erev 0 \
        --v-init -60 \
        --syn-weight 1 \
        -m psp


