export DATASET_PREFIX=$SCRATCH/striped/dentate
export DATA_PREFIX=./datasets
export DATASET_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

python3  network_clamp.py go -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
         --template-paths templates \
         -p HC -g 1032250 -t 9500 --dt 0.025 \
         --dataset-prefix $DATASET_PREFIX \
         --config-prefix config \
         --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag --phase-mod \
         --results-path results/netclamp \
         --params-path results/network_clamp.optimize.HC_20210804_013731.yaml

