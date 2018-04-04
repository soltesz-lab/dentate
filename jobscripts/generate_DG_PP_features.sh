#!/bin/bash

mpirun.mpich -np 4 python ./scripts/generate_DG_PP_features.py \
   --config=./config/Full_Scale_Control.yaml \
   --forest-path=./datasets/AAC_forest_syns_20171102.h5 \
   --connectivity-path=./datasets/Test_GC_1000/AAC_test_connections_20180323.h5 \
   --connectivity-namespace=Connections \
   --coords-path=./datasets/dentate_Full_Scale_Control_coords_20180214.h5 \
   --coords-namespace="Generated Coordinates" \
   --io-size=1 --resample-volume=2 -v
