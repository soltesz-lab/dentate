#!/bin/bash

mpirun.mpich -np 1 python3 ./scripts/generate_distance_connections.py \
             --config=Full_Scale_GC_Exc_Sat_SynExp3NMDA2fd_CLS_IN_PR.yaml \
             --config-prefix=./config \
             --forest-path=./datasets/Single/data_601886.h5 \
             --connectivity-path=./datasets/Single/data_601886.h5 \
             --connectivity-namespace=Connections \
             --coords-path=./datasets/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
             --coords-namespace=Coordinates \
             --io-size=1 -v
