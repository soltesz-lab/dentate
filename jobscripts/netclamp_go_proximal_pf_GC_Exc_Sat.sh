#!/bin/bash


export dataset_prefix=./datasets
#export dataset_prefix=$WORK/dentate


# 118936 results/netclamp/GC_20220926/optimize_selectivity.20220928_025352.yaml
# 128708 results/netclamp/optimize_selectivity.gid_128708.20220926_101452.yaml
# 239049 results/netclamp/optimize_selectivity.gid_239049.20220926_102806.yaml
# 738162 results/netclamp/optimize_selectivity.gid_738162.20220926_102311.yaml

mpirun -n 1 python3 network_clamp.py go \
       -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR_proximal_pf_gid_118936.yaml \
         -p GC -g 118936 -t 9450 --dt 0.025 --use-coreneuron \
         --template-paths templates \
         --dataset-prefix $dataset_prefix \
         --input-features-path $dataset_prefix/Full_Scale_Control/DG_input_features_20220216.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag  --n-trials 1 \
         --recording-profile 'Network clamp all synaptic' \
         --config-prefix config  \
         --results-path results/netclamp/runs

