#!/bin/bash

mpirun.mpich -np 8  python ./scripts/generate_DG_PP_spiketrains.py \
   --config=./config/Full_Scale_Pas.yaml \
   --features-path=./datasets/DG_PP_features_100_20190501.h5 \
   --output-path=./datasets/DG_PP_spiketrains_100_20190501.h5 \
   --stimulus-id=100 \
   --io-size=2 -v 


