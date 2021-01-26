#ibrun -n 1 python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml --template-paths templates \
yamlfilename=$1

python3  network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml --template-paths templates \
         -p MC -g 1015624 -t 9500 --dt 0.001 \
         --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
         --config-prefix config \
         --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag \
         --results-path results/netclamp \
         --params-path /scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/$yamlfilename
