export DATASET_PREFIX=$SCRATCH/striped/dentate

mpirun -n 1 python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
       --template-paths templates \
       -p BC -g 1039000 -t 9500 --dt 0.001 \
       --dataset-prefix $DATASET_PREFIX \
       --config-prefix config \
       --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
       --input-features-namespaces 'Place Selectivity' \
       --input-features-namespaces 'Grid Selectivity' \
       --input-features-namespaces 'Constant Selectivity' \
       --arena-id A --trajectory-id Diag \
       --results-path results/netclamp \
       --params-path results/netclamp/network_clamp.optimize.BC_1039000_20201027_164832.yaml
