#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
   --config=Full_Scale_GC_Exc_Sat.yaml \
   --forest-path=./datasets/DG_MC_forest_syns_20190426.h5 \
   --connectivity-path=./datasets/MC_test_connections_20190426.h5 \
   --connectivity-namespace=Connections \
   --coords-path=./datasets/DG_coords_20180703.h5 \
   --coords-namespace="Coordinates" \
   --io-size=2 --dry-run -v


