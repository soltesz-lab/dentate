#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
             --config=Full_Scale_Pas.yaml \
             --config-prefix=./config \
             --forest-path=./datasets/DGC_forest_test_syns_20181222.h5 \
             --connectivity-path=./datasets/Small_Scale/GC_test_connections_20181222.h5 \
             --connectivity-namespace=Connections \
             --coords-path=./datasets/DG_coords_20180717.h5 \
             --coords-namespace=Coordinates \
             --io-size=2 -v
