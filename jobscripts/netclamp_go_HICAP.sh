export DATA_PREFIX=/media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG

python3 network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR.yaml \
         -p HCC -g 1043250 -t 9500 --dt 0.01 --use-coreneuron \
         --dataset-prefix $DATA_PREFIX \
         --template-paths templates:$HOME/src/model/DGC/Mateos-Aparicio2014:$HOME/src/model/XPPcode \
         --config-prefix config \
         --input-features-path $DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --arena-id A --trajectory-id Diag \
         --results-path results/netclamp


