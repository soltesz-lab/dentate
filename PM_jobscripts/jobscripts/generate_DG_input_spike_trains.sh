#!/bin/bash

dataset_prefix=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/Full_Scale_Control

mpirun.mpich -np 2 python3 ./scripts/generate_input_spike_trains.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --selectivity-path=${dataset_prefix}/DG_input_features_20200320.h5 \
             --output-path=${dataset_prefix}/DG_input_spike_trains_20200321.h5 \
             --n-trials=3 --dry-run \
             -p MPP -p LPP -p MC -v --gather --plot --save-fig DG_input_spike_trains


