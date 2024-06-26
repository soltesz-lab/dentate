export DATASET_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

mpirun -n 1 python3 network_clamp.py go -c Network_Clamp_GC_Aradi_SLN_IN_PR.yaml \
       -p BC -g 1042795 -t 9500 --dt 0.0125 --use-coreneuron \
       --dataset-prefix $DATASET_PREFIX \
       --template-paths templates \
       --config-prefix config \
       --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
       --input-features-namespaces 'Place Selectivity' \
       --input-features-namespaces 'Grid Selectivity' \
       --input-features-namespaces 'Constant Selectivity' \
       --arena-id A --trajectory-id Diag \
       --results-path results/netclamp

