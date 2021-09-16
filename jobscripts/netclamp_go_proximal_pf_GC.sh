#!/bin/bash

export dataset_prefix=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

python3 network_clamp.py go  -c Network_Clamp_GC_Aradi_Sat_SLN_IN_PR_proximal_pf.yaml \
         -p GC -g 505461  -t 9400 --use-coreneuron --dt 0.01 \
         --template-paths templates:$HOME/src/model/DGC/Aradi1999 \
         --dataset-prefix $dataset_prefix \
         --input-features-path $dataset_prefix/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag  --n-trials 1 --phase-mod \
         --coords-path ./datasets/DG_coords_20190717_compressed.h5 \
         --recording-profile 'Network clamp default' \
         --config-prefix config  \
         --results-path results/netclamp

