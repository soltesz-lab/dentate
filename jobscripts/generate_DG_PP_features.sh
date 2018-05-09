#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_DG_PP_features.py \
   --config=./config/Full_Scale_Control.yaml \
   --stimulus-id=100 \
   --coords-path=./datasets/DG_cells_20180305.h5 \
   --output-path=./datasets/DG_PP_features_100_20180501.h5 \
   --io-size=1 -v 
