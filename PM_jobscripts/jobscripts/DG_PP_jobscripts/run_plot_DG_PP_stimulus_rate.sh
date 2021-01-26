#!/bin/bash

python -i scripts/plot_stimulus_rate.py --features-path='datasets/Full_Scale_Control/DG_PP_spikes_101718.h5' \
    --features-namespace='Vector Stimulus' --trajectory-id='100' --include=MPP --show-fig

#python -i scripts/plot_stimulus_rate.py --features-path='datasets/Full_Scale_Control/DG_PP_spikes_101718.h5' \
#    --features-namespace='Vector Stimulus' --trajectory-id='100' --include=LPP --show-fig



