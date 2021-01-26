#!/bin/bash

python -i scripts/plot_DG_PP_metrics.py --coords-path='datasets/Full_Scale_Control/DG_coords_20180717.h5' \
    --features-path='datasets/Full_Scale_Control/DG_PP_features_101718.h5' --distances-namespace='Arc Distances' \
    --population='MPP' --cell-type='grid' --normed=0 --show-fig=1 --save-fig=0 --bin-size=300
