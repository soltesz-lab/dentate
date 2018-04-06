#!/bin/bash

mpirun.mpich -np 4 python ./scripts/generate_DG_PP_features.py \
   --config=./config/Full_Scale_Control.yaml \
   --stimulus-id=110 \
   --coords-path=./datasets/DG_cells_20180305.h5 \
   --output-path=./datasets/DG_PP_features_110_20180404.h5 \
   --io-size=1 -v 
