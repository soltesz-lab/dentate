#!/bin/bash

dataset_prefix=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/Full_Scale_Control

mpirun.mpich -np 8 python3 ./scripts/generate_DG_input_spike_trains.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --selectivity-path=${dataset_prefix}/DG_input_features_20190724.h5 \
             --output-path=${dataset_prefix}/DG_input_spike_trains_20190724.h5 \
             -p MPP -p LPP -p CA3c -v --gather --plot --save-fig DG_input_spike_trains


