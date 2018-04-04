#!/bin/bash

mpirun.mpich -np 4 python ./scripts/generate_DG_PP_spiketrains.py \
   --config=./config/Full_Scale_Control.yaml \
   --features-path=./datasets/DG_PP_features_110_20180404.h5 \
   --stimulus-id=110 \
   --io-size=1 -v 


