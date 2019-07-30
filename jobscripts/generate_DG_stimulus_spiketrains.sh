#!/bin/bash

for trajectory in HDiag Diag DiagL5 DiagU5; do
    
mpirun.mpich -np 8  python ./scripts/generate_DG_stimulus_spike_trains.py \
    --config=Full_Scale_Pas.yaml \
    --features-path=./datasets/DG_stimulus_20190610.h5 \
    --output-path=./datasets/DG_stimulus_20190610.h5 \
    --arena-id=A \
    --trajectory-id=${trajectory} \
    -p LPP -p MPP \
    --io-size=2 -v

done
