#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
  --config=./config/Full_Scale_Control.yaml \
  --forest-path=./datasets/Small_Scale/GC_MC_BC_trees_20181126.h5 \
  --connectivity-path=./datasets/Small_Scale/GC_MC_BC_connections_20181126.h5 \
  --connectivity-namespace=Connections \
  --coords-path=./datasets/DG_coords_20180717.h5 \
  --coords-namespace=Coordinates \
  --io-size=2 -v
