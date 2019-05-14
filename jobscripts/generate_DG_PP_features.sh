#!/bin/bash

mpirun.mpich -np 1 python ./scripts/generate_DG_PP_features.py \
             --config=Full_Scale_Pas.yaml \
             --config-prefix=./config \
             --coords-path=./datasets/DG_coords_20190122.h5 \
             --output-path=./datasets/DG_PP_features_A_20190512.h5 \
             --io-size=1 -v 
