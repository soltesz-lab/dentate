#!/bin/bash

export DATA_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

for gid in 1043250 1043600 1043950 1044300 1044649; do
    
    mpirun -n 1 python3 ./cell_clamp.py \
           -g $gid \
           --population HCC \
           --config-prefix config \
           --config-file 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
           --dataset-prefix $DATA_PREFIX \
           --template-paths=$HOME/model/DGC/Mateos-Aparicio2014:templates \
           --results-path=results/cell_clamp \
           --presyn-name GC \
           --syn-mech-name AMPA \
           --erev 0 \
           --v-init -60 \
           --syn-weight 1 \
           -m psp
    
    mpirun -n 1 python3 ./cell_clamp.py \
           -g $gid \
           --population HCC \
           --config-prefix config \
           --config-file 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
           --dataset-prefix $DATA_PREFIX \
           --template-paths=$HOME/model/DGC/Mateos-Aparicio2014:templates \
           --results-path=results/cell_clamp \
           --presyn-name MC \
           --syn-mech-name AMPA \
           --erev 0 \
           --v-init -60 \
           --syn-weight 1 \
           -m psp
    
    mpirun -n 1 python3 ./cell_clamp.py \
           -g $gid \
           --population HCC \
           --config-prefix config \
           --config-file 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
           --dataset-prefix $DATA_PREFIX \
           --template-paths=$HOME/model/DGC/Mateos-Aparicio2014:templates \
           --results-path=results/cell_clamp \
           --presyn-name CA3c \
           --syn-mech-name AMPA \
           --erev 0 \
           --v-init -60 \
           --syn-weight 1 \
           -m psp
    
    mpirun -n 1 python3 ./cell_clamp.py \
           -g $gid \
           --population HCC \
           --config-prefix config \
           --config-file 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
           --dataset-prefix $DATA_PREFIX \
           --template-paths=$HOME/model/DGC/Mateos-Aparicio2014:templates \
           --results-path=results/cell_clamp \
           --presyn-name MPP \
           --syn-mech-name AMPA \
           --erev 0 \
           --v-init -60 \
           --syn-weight 1 \
           -m psp

    

done

