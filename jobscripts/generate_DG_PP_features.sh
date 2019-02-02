#!/bin/bash

mpirun.mpich -np 1 python ./scripts/generate_DG_PP_features.py \
   --config=./config/Full_Scale_Pas.yaml \
   --input-params-file-path=./config/Input_Features.yaml \
   --stimulus-id=100 \
   --coords-path=./datasets/DG_coords_20180717.h5 \
   --output-path=./datasets/DG_PP_features_100_20190131.h5 \
   --io-size=1 -v 
