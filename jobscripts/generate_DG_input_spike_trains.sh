#!/bin/bash

#dataset_prefix=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/Full_Scale_Control
dataset_prefix=./datasets

mpirun.mpich -np 2 python3 ./scripts/generate_input_spike_trains.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --selectivity-path=${dataset_prefix}/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
             --output-path=${dataset_prefix}/DG_input_spike_trains_test_20210605.h5 \
             --n-trials=1 --dry-run --phase-mod --debug --coords-path ${dataset_prefix}/DG_coords_20190717_compressed.h5 \
             -p MPP -v --gather --plot --save-fig DG_input_spike_trains --dry-run


