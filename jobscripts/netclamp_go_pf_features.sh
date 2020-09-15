#!/bin/bash

export dataset_prefix=./datasets
python3  network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
         -p GC -g 317622  -t 9400 \
         --template-paths templates:$HOME/src/model/DGC/Mateos-Aparicio2014 \
         --dataset-prefix $dataset_prefix \
         --input-features-path $dataset_prefix/DG_input_features_20200901_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag  --n-trials 1 \
         --recording-profile 'Network clamp default' \
         --config-prefix config  \
         --results-path results/netclamp

