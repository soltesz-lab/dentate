python3  network_clamp.py go -c Network_Clamp_GC_Exc_Sat_DD_SLN.yaml --template-paths templates \
-p AAC -g 1042800 -t 5000 --dataset-prefix /media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG \
         --config-prefix config -s /media/igr/d865f900-7fcd-45c7-a7a7-bd2a7391bc40/Data/DG/results/Full_Scale_GC_Exc_Sat_DD_SLN_Diag_157780/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_results_compressed.h5 \
         --results-path results/netclamp 
