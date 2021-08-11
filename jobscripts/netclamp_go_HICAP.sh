export DATA_PREFIX=$SCRATCH/striped/dentate

ibrun -n 1 python3  network_clamp.py go -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
         --template-paths templates \
         -p HCC -g 1043250 -t 9500 --dt 0.001 \
         --dataset-prefix $DATA_PREFIX \
         --config-prefix config \
         --input-features-path $DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag \
         --results-path results/netclamp --use-coreneuron --phase-mod \
         --params-path 

