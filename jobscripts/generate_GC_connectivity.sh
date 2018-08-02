#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
  --config=./config/Full_Scale_Control.yaml \
  --forest-path=./datasets/Test_GC_1000/DGC_forest_test_syns_20180801.h5 \
  --connectivity-path=./datasets/Test_GC_1000/DG_GC_test_connections_20180801.h5 \
  --connectivity-namespace=Connections \
  --coords-path=./datasets/DG_coords_20180717.h5 \
  --coords-namespace=Coordinates \
  --io-size=2 -v
