#export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
#export FI_MLX_ENABLE_SPAWN=yes
export DATA_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data

mpirun -n 8 python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p HCC -g 1043250 -t 9500 \
    --dataset-prefix $DATA_PREFIX/DG \
    --config-prefix config \
    --input-features-path $DATA_PREFIX/DG/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp \
    --param-config-name "Weight inh" \
    --cooperative-init \
    --opt-iter 400 rate
