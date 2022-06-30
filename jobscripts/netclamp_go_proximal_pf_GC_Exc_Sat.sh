#!/bin/bash


export dataset_prefix=./datasets
export dataset_prefix=$WORK/dentate


# 118936
# 253724
# 113699
# 752854

ibrun -n 1 python3 network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR_proximal_pf_gid_118936.yaml \
         -p GC -g 118936 -t 9500 --dt 0.025 \
         --template-paths templates \
         --dataset-prefix $dataset_prefix \
         --input-features-path $dataset_prefix/Full_Scale_Control/DG_input_features_20220216.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag  --n-trials 1 \
         --recording-profile 'Network clamp all synaptic' \
         --config-prefix config  \
         --params-path $dataset_prefix/results/optimize_selectivity.20220605_152702.yaml \
         --params-id 0 \
         --results-path results/netclamp

