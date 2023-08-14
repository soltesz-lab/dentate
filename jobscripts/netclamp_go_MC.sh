export dataset_prefix=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/
export dataset_prefix=datasets

mpirun -n 1 python3  network_clamp.py go -c Network_Clamp_GC_Exc_Sat_SynExp3NMDA2_SLN_CLS_IN_PR_center_pf.yaml \
       --template-paths templates --dt 0.0125 --use-coreneuron \
       -p MC -g 1015662 -t 9500 \
       --dataset-prefix $dataset_prefix \
       --config-prefix config \
       --input-features-path $dataset_prefix/Full_Scale_Control/DG_input_features_20220216.h5 \
       --input-features-namespaces 'Place Selectivity' \
       --input-features-namespaces 'Grid Selectivity' \
       --input-features-namespaces 'Constant Selectivity' \
       --recording-profile 'Network clamp all synaptic' \
       --arena-id A --trajectory-id Diag \
       --results-path results/netclamp/runs \
       --params-id 0 \
       --params-path results/netclamp/MC_20230204/optimize_selectivity.20230204_192554.yaml



