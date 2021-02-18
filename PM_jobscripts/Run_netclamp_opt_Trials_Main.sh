export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
" HC 1030000 15879716 "
" HC 1032250 39786206 "
" HC 1034500 17324444 "
" HC 1036750 96962730 "
" HC 1038999 69758687 "
" BC 1039000 93454042 "
" BC 1039950 27431042 "
" BC 1040900 32603857 "
" BC 1041850 87742037 "
" BC 1042799 183453 "
" AAC 1042800 4893658 "
" AAC 1042913 291547 "
" AAC 1043025 62387369 "
" AAC 1043138 19281943 "
" AAC 1043249 54652217 "
" IS 1049650 4259860 "
" IS 1050400 63796673 "
" IS 1051150 22725943 "
" IS 1051900 82185961 "
" IS 1052649 45707385 "
)
lif=(
" HCC 1043250 33236209 "
" HCC 1043600 42402504 "
" HCC 1043950 99434948 "
" HCC 1044300 97569363 "
" HCC 1044649 84121621 "
" NGFC 1044650 12740157 "
" NGFC 1045900 67428613 "
" NGFC 1047150 59071557 "
" NGFC 1048400 80347840 "
" NGFC 1049649 77179804 "
" MOPP 1052650 31571230 "
" MOPP 1053650 35281577 "
" MOPP 1054650 60567645 "
" MOPP 1055650 34097792 "
" MOPP 1056649 17666981 "
)

pil=(
" HC 1030000 15879716 "
" HC 1030000 45419272 "
" HC 1030000 53736785 "
" HC 1030000 63599789 "
" HC 1032250 39786206 "
" BC 1042799 67308587 "
" AAC 1043249 26662642 "
" HCC 1043600 42402504 "
" MOPP 1053650 78038978 "
)

N_cores=8

IFS='
'
counter=0
for f in ${fil[@]}
do

set -- "$f" 
IFS=" " ; declare -a tempvar=($*) 


#ibrun -n 56 -o  0 task_affinity ./mycode.exe input1 &   # 56 tasks; offset by  0 entries in hostfile.
#ibrun -n 56 -o 56 task_affinity ./mycode.exe input2 &   # 56 tasks; offset by 56 entries in hostfile.
#wait                                                    # Required; else script will exit immediately.


#ibrun -n $N_cores -o $counter python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
#    --template-paths templates \
#    -p ${tempvar[0]:1:-1} -g ${tempvar[1]} -t 9500 --dt 0.001 \
#    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
#    --config-prefix config \
#    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \
#    --arena-id A --trajectory-id Diag \
#    --results-path results/netclamp \
#    --opt-seed ${tempvar[3]} \
#    --params-path ${tempvar[2]:1:-1} &


#pop=${tempvar[0]:1:-1}
pop=${tempvar[0]}
gid=${tempvar[1]}
seed=${tempvar[2]}

#ibrun -n 8 python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
ibrun -n $N_cores -o $((counter * 14))  python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p $pop -g $gid -t 9500 --dt 0.001 \
    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
    --config-prefix config \
    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp \
    --n-trials 4 \
    --param-config-name "Weight exc inh microcircuit" \
    --opt-seed $seed \
    --opt-iter 400 rate & 


#    --results-file network_clamp.optimize.$pop\_$gid\_$(date +'%Y%m%d_%H%M%S')\_$seed.h5 \

counter=$((counter + 1))

done
wait
