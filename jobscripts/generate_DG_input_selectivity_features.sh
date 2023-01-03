#!/bin/bash

dataset_prefix=./datasets/Full_Scale_Control
#             -p LPP -p MPP -p MC -p CA3c \

mpirun.mpich -np 6 python3 ./scripts/generate_input_selectivity_features.py \
             -p LPP --use-noise-gen \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --coords-path=${dataset_prefix}/DG_coords_20190717_compressed.h5 \
             --output-path=${dataset_prefix}/DG_input_features_test_20220216.h5 \
             -v --gather --debug --debug-count 1000  --plot # --save-fig DG_input_selectivity_features 

