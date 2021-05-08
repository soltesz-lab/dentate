export DATASET_PREFIX=$SCRATCH/striped/dentate

ibrun -n 1 python3  network_clamp.py go -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
         --template-paths templates \
         -p HC -g 1030000 -t 9500 --dt 0.001 --use-coreneuron \
         --dataset-prefix $DATASET_PREFIX \
         --config-prefix config \
         --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag \
         --results-path results/netclamp \
         --params-path results/network_clamp.optimize.HC_20210311_132602_15879716.yaml

