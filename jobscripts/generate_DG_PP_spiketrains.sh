#!/bin/bash

for trajectory in HDiag Diag DiagL5 DiagU5; do
    
mpirun.mpich -np 8  python ./scripts/generate_DG_PP_spiketrains.py \
    --config=Full_Scale_Pas.yaml \
    --features-path=./datasets/DG_PP_features_A_20190512.h5 \
    --output-path=./datasets/DG_PP_spiketrains_A_20190512.h5 \
    --arena-id=A \
    --trajectory-id=${trajectory} \
    --io-size=2 -v

done
