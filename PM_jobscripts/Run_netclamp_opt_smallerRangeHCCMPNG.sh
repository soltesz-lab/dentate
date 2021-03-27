export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
" HCC 1043250 71407528 "
)
lif=(
" HCC 1043250 17609813 "
" HCC 1043250 12260638 "
" HCC 1043600 42402504 "
" HCC 1043600 89433777 "
" HCC 1043600 60991105 "
" HCC 1043600 64290895 "
" HCC 1043600 17293770 "
" HCC 1043950 99434948 "
" HCC 1043950 57660249 "
" HCC 1043950 54593731 "
" HCC 1043950 72125941 "
" HCC 1043950 41566230 "
" HCC 1044300 97569363 "
" HCC 1044300 66834161 "
" HCC 1044300 94613541 "
" HCC 1044300 63592626 "
" HCC 1044300 92910319 "
" HCC 1044649 84121621 "
" HCC 1044649 94560988 "
" HCC 1044649 46366417 "
" HCC 1044649 24805208 "
" HCC 1044649 59396015 "
" NGFC 1044650 12740157 "
" NGFC 1044650 97895890 "
" NGFC 1044650 93872787 "
" NGFC 1044650 95844113 "
" NGFC 1044650 96772370 "
" NGFC 1045900 67428613 "
" NGFC 1045900 95436908 "
" NGFC 1045900 6112188 "
" NGFC 1045900 71039025 "
" NGFC 1045900 89814943 "
" NGFC 1047150 59071557 "
" NGFC 1047150 77901687 "
" NGFC 1047150 27400566 "
" NGFC 1047150 50965365 "
" NGFC 1047150 48744644 "
" NGFC 1048400 80347840 "
" NGFC 1048400 38650070 "
" NGFC 1048400 62046131 "
" NGFC 1048400 35472841 "
" NGFC 1048400 35297351 "
" NGFC 1049649 77179804 "
" NGFC 1049649 26628153 "
" NGFC 1049649 99082330 "
" NGFC 1049649 89481705 "
" NGFC 1049649 10249569 "
" MOPP 1052650 31571230 "
" MOPP 1052650 45373570 "
" MOPP 1052650 85763600 "
" MOPP 1052650 68839073 "
" MOPP 1052650 29079471 "
" MOPP 1053650 35281577 "
" MOPP 1053650 82093235 "
" MOPP 1053650 78038978 "
" MOPP 1053650 39888091 "
" MOPP 1053650 59550066 "
" MOPP 1054650 60567645 "
" MOPP 1054650 94967765 "
" MOPP 1054650 21247851 "
" MOPP 1054650 26628185 "
" MOPP 1054650 3611780 "
" MOPP 1055650 34097792 "
" MOPP 1055650 44866707 "
" MOPP 1055650 61810606 "
" MOPP 1055650 83145544 "
" MOPP 1055650 79924848 "
" MOPP 1056649 17666981 "
" MOPP 1056649 88486608 "
" MOPP 1056649 92808036 "
" MOPP 1056649 73504121 "
" MOPP 1056649 68347478 "
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
    --param-config-name "Weight exc inh microcircuit" \
    --opt-seed $seed \
    --opt-iter 400 rate & 


#    --results-file network_clamp.optimize.$pop\_$gid\_$(date +'%Y%m%d_%H%M%S')\_$seed.h5 \

counter=$((counter + 1))

done
wait
