#!/bin/bash

mpirun.mpich -np 1 python ./scripts/generate_DG_stimulus_features.py \
             --config=Full_Scale_Pas.yaml \
             --config-prefix=./config \
             --coords-path=./datasets/DG_coords_20190521.h5 \
             --output-path=./datasets/DG_stimulus_20190610.h5 \
             -p LPP -p MPP -v 
