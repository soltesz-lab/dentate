#export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_core.so:/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes

ibrun -n 8 python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p MOPP -g 1052650 -t 9500 \
    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
    --config-prefix config \
    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path $SCRATCH/model/dentate/results/netclamp \
    --param-config-name "Weight inh" \
    --opt-iter 400 rate
