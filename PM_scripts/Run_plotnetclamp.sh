export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1056649_20210212_002016_92808036.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1056649_20210212_002016_88486608.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1056649_20210212_002016_73504121.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1056649_20210212_002016_68347478.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1056649_20210212_002016_17666981.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1049649_20210212_002016_99082330.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1049649_20210212_002016_89481705.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1049649_20210212_002016_77179804.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1049649_20210212_002016_26628153.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1049649_20210212_002016_10249569.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1048400_20210212_002016_80347840.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1055650_20210212_002016_83145544.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1055650_20210212_002016_79924848.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1055650_20210212_002016_61810606.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1055650_20210212_002016_44866707.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1055650_20210212_002016_34097792.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1048400_20210212_002016_62046131.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1048400_20210212_002016_38650070.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1048400_20210212_002016_35472841.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1048400_20210212_002016_35297351.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1053650_20210212_002016_82093235.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1053650_20210212_002016_78038978.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1053650_20210212_002016_59550066.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1053650_20210212_002016_39888091.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1047150_20210212_002016_77901687.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1047150_20210212_002016_59071557.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1047150_20210212_002016_50965365.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1047150_20210212_002016_48744644.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1047150_20210212_002016_27400566.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1045900_20210212_002016_71039025.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1045900_20210212_002016_67428613.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20210212_002016_97895890.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20210212_002016_96772370.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20210212_002016_95844113.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20210212_002016_12740157.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1054650_20210212_002016_94967765.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1054650_20210212_002016_26628185.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1054650_20210212_002016_21247851.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1054650_20210212_002016_03611780.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1053650_20210212_002016_35281577.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1045900_20210212_002016_95436908.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1045900_20210212_002016_89814943.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1045900_20210212_002016_06112188.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20210212_002016_93872787.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1052650_20210212_002016_85763600.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1052650_20210212_002016_68839073.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1052650_20210212_002016_45373570.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1052650_20210212_002016_31571230.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_1052650_20210212_002016_29079471.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044649_20210212_002016_94560988.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044649_20210212_002016_46366417.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044649_20210212_002016_84121621.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044649_20210212_002016_59396015.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044649_20210212_002016_24805208.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044300_20210212_002016_97569363.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044300_20210212_002016_92910319.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043950_20210212_002016_99434948.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043950_20210212_002016_54593731.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043950_20210212_002016_41566230.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044300_20210212_002016_94613541.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044300_20210212_002016_66834161.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1044300_20210212_002016_63592626.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043950_20210212_002016_72125941.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043950_20210212_002016_57660249.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043600_20210212_002016_64290895.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043600_20210212_002016_89433777.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043600_20210212_002016_60991105.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043600_20210212_002016_42402504.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043600_20210212_002016_17293770.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043250_20210212_002016_92055940.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043250_20210212_002016_71407528.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043250_20210212_002016_17609813.h5 "
"  results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043250_20210212_002016_12260638.h5 "
)

newfil=(
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_023909_53499406.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_023711_36010476.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_023711_27137089.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_024103_49937004.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_023516_04893658.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210525_140015_network_clamp.optimize.BC_20210525_032710_52357252.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210525_140015_network_clamp.optimize.BC_20210525_025416_74865768.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210525_140015_network_clamp.optimize.BC_20210525_025416_01503844.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210525_140015_network_clamp.optimize.BC_20210525_032714_28135771.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210525_140015_network_clamp.optimize.BC_20210525_024631_93454042.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210525_140852_network_clamp.optimize.HC_20210525_033045_45419272.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210525_140852_network_clamp.optimize.HC_20210525_033045_15879716.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210525_140852_network_clamp.optimize.HC_20210525_033439_63599789.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210525_140852_network_clamp.optimize.HC_20210525_033439_53736785.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210525_140852_network_clamp.optimize.HC_20210525_033427_28682721.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_20210525_141603_network_clamp.optimize.HCC_20210525_032714_17609813.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_20210525_141603_network_clamp.optimize.HCC_20210525_032854_92055940.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_20210525_141603_network_clamp.optimize.HCC_20210525_032853_71407528.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_20210525_141603_network_clamp.optimize.HCC_20210525_033427_12260638.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_20210525_141603_network_clamp.optimize.HCC_20210525_032714_33236209.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_20210525_142243_network_clamp.optimize.IS_20210525_055113_84013649.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_20210525_142243_network_clamp.optimize.IS_20210525_054954_75940072.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_20210525_142243_network_clamp.optimize.IS_20210525_055045_49627038.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_20210525_142243_network_clamp.optimize.IS_20210525_054948_11745958.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_20210525_142243_network_clamp.optimize.IS_20210525_054514_04259860.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210525_142655_network_clamp.optimize.MOPP_20210525_044615_29079471.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210525_142655_network_clamp.optimize.MOPP_20210525_044040_31571230.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210525_142655_network_clamp.optimize.MOPP_20210525_044357_68839073.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210525_142655_network_clamp.optimize.MOPP_20210525_044321_85763600.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210525_142655_network_clamp.optimize.MOPP_20210525_044054_45373570.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_20210525_143341_network_clamp.optimize.NGFC_20210525_053539_96772370.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_20210525_143341_network_clamp.optimize.NGFC_20210525_050031_93872787.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_20210525_143341_network_clamp.optimize.NGFC_20210525_053536_95844113.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_20210525_143341_network_clamp.optimize.NGFC_20210525_045919_97895890.h5 "
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_20210525_143341_network_clamp.optimize.NGFC_20210525_045136_12740157.h5 "
)

testgid=(
" results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_AAC_20210525_135736_network_clamp.optimize.AAC_20210525_023909_53499406.h5 "
)

N_cores=1

IFS='
'
counter=0
for f in ${testgid[@]}
do

set -- "$f" 
IFS=" " ; declare -a tempvar=($*) 


preseed=${f: -12}
seed=${preseed:0:8}
#ibrun -n 56 -o  0 task_affinity ./mycode.exe input1 &   # 56 tasks; offset by  0 entries in hostfile.
#ibrun -n 56 -o 56 task_affinity ./mycode.exe input2 &   # 56 tasks; offset by 56 entries in hostfile.
#wait                                                    # Required; else script will exit immediately.

#pop=${tempvar[0]:1:-1}
#pop=${tempvar[0]}
#gid=${tempvar[1]}
#seed=${tempvar[2]}
#yaml=${tempvar[3]}
#
ibrun -n $N_cores -o $((counter)) python3 PM_scripts/plot_network_clamp.py \
        -p $f --opt-seed $seed & 
#    --template-paths templates \
#    -p $pop -g $gid -t 9500 --dt 0.001 \
#    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
#    --config-prefix config \
#    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \
#    --arena-id A --trajectory-id Diag \
#    --results-path results/netclamp \
#    --opt-seed $seed \
#    --params-path $yaml &



##ibrun -n 8 python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
#ibrun -n $N_cores -o $((counter * 56))  python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
#    --template-paths templates \
#    -p $pop -g $gid -t 9500 --dt 0.001 \
#    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
#    --config-prefix config \
#    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \
#    --arena-id A --trajectory-id Diag \
#    --results-path results/netclamp \
#    --param-config-name "Weight exc inh microcircuit" \
#    --opt-seed $seed \
#    --opt-iter 400 rate & 


#    --results-file network_clamp.optimize.$pop\_$gid\_$(date +'%Y%m%d_%H%M%S')\_$seed.h5 \

counter=$((counter + 1))

done
wait
