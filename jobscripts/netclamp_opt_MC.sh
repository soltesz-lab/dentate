export DATASET_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

mpirun.mpich -n 8 python3  network_clamp.py optimize -c Network_Clamp_GC_Aradi_SLN_IN_PR.yaml \
    --template-paths templates --dt 0.0125 --use-coreneuron \
    -p MC -g 1015624 -t 9500 \
    --dataset-prefix $DATASET_PREFIX \
    --config-prefix config \
    --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp \
    --param-config-name "Weight exc inh microcircuit" \
    --opt-iter 600 rate
