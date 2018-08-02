#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
   --config=./config/Full_Scale_Control.yaml \
   --resolution 30 30 10 \
   --forest-path=./datasets/MC_forest_syns_20180706.h5 \
   --connectivity-path=./datasets/MC_test_connections_20180706.h5 \
   --connectivity-namespace=Connections \
   --coords-path=./datasets/DG_coords_20180703.h5 \
   --coords-namespace="Coordinates" \
   --io-size=2 -v
