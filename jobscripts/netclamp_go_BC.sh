export DATASET_PREFIX=$SCRATCH/striped/dentate
export DATASET_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

mpirun -n 1 python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
       --template-paths templates \
       -p BC -g 1039000 -g 1039001 -t 9500 --dt 0.025 --use-coreneuron \
       --dataset-prefix $DATASET_PREFIX \
       --config-prefix config \
       --input-features-path $DATASET_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
       --input-features-namespaces 'Place Selectivity' \
       --input-features-namespaces 'Grid Selectivity' \
       --input-features-namespaces 'Constant Selectivity' \
       --arena-id A --trajectory-id Diag \
       --results-path results/netclamp \
       --params-path config/20201105_Izhi_compiled_BC.yaml

