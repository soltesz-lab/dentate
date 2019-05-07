#!/bin/bash

mpirun.mpich -np 8  python ./scripts/generate_DG_PP_spiketrains.py \
   --config=./config/Full_Scale_Pas.yaml \
   --features-path=./datasets/DG_PP_features_A_20190501.h5 \
   --output-path=./datasets/DG_PP_spiketrains_A_Diag_20190501.h5 \
   --arena-id=A \
   --trajectory-id=Diag \
   --io-size=2 -v


