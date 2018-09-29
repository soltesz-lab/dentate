#!/bin/bash

mpirun.mpich -np 8  python ./scripts/generate_DG_PP_spiketrains.py \
   --config=./config/Full_Scale_Control.yaml \
   --features-path=./datasets/DG_PP_features_20180928.h5 \
   --output-path=./datasets/DG_PP_spiketrains_100_20180928.h5 \
   --stimulus-id=100 \
   --io-size=1 -v 


