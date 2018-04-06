#!/bin/bash

mpirun.mpich -np 4 python ./scripts/generate_distance_connections.py \
  --config=./config/Full_Scale_Control.yaml \
  --forest-path=./datasets/DGC_forest_test_syns_20180322.h5 \
  --connectivity-path=./datasets/Test_GC_1000/DG_GC_test_connections_20180402.h5 \
  --connectivity-namespace=Connections \
  --coords-path=./datasets/DG_Cells_Full_Scale_20180326.h5 \
  --coords-namespace=Coordinates \
  --io-size=2 --resample-volume=1 -v
