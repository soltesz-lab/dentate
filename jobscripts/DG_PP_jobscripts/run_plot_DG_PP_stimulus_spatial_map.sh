#!/bin/bash

python -i scripts/plot_stimulus_spatial_map.py --features-path='datasets/Full_Scale_Control/DG_PP_spikes_101718.h5' \
    --features-namespace='Vector Stimulus 100' --coords-path='datasets/Full_Scale_Control/DG_coords_20180717.h5' \
    --distances-namespace='Arc Distances' --include='MPP' --include='LPP' --trajectory-id='100' --bin-size=300 \
    --from-spikes=False --show-fig

