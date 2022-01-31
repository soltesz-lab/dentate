#!/bin/bash

dataset_prefix=./datasets/Full_Scale_Control
#             -p LPP -p MPP -p MC \

mpirun.mpich -np 1 python3 ./scripts/generate_input_selectivity_features.py \
             -p MC \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --coords-path=${dataset_prefix}/DG_coords_20190717_compressed.h5 \
             --output-path=${dataset_prefix}/DG_input_features_20200320.h5 \
             -v --gather --plot --save-fig DG_input_selectivity_features --dry-run --debug

