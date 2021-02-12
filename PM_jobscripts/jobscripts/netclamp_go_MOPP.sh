ibrun -n 1 python3  network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml --template-paths templates \
         -p MOPP -g 1052650 -t 9500 \
         --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
         --config-prefix config \
         --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag \
         --results-path results/netclamp \
         --params-path /scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/network_clamp.optimize.80403730-9c80-40c4-9981-bcbf38188664.yaml

#         -s /media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/results/Full_Scale_GC_Exc_Sat_DD_SLN_Diag_155005/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_results_compressed.h5 \
#         --recording-profile "Network clamp exc synaptic" \
