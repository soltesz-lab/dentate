#!/bin/bash

dataset_prefix=./datasets/Full_Scale_Control
#             -p LPP -p MPP -p MC \

nohup mpirun.mpich -np 8 python3 ./scripts/generate_input_selectivity_features.py \
             -p MC -p LPP -p MPP -p CA3c \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --coords-path=${dataset_prefix}/DG_coords_20190717_compressed.h5 \
             --output-path=${dataset_prefix}/DG_input_features_20220131.h5 \
             -v --gather --plot --save-fig DG_input_selectivity_features &

