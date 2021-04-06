export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
" HCC 1043250 12260638 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_12260638.yaml "
)
lif=(
" HCC 1043250 17609813 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_17609813.yaml "
" HCC 1043250 33236209 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_33236209.yaml "
" HCC 1043250 71407528 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_71407528.yaml "
" HCC 1043250 92055940 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_92055940.yaml "
" HCC 1043600 17293770 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_144349_17293770.yaml "
" HCC 1043600 42402504 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_144349_42402504.yaml "
" HCC 1043600 60991105 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_144349_60991105.yaml "
" HCC 1043600 64290895 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_144349_64290895.yaml "
" HCC 1043600 89433777 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_144349_89433777.yaml "
" HCC 1043950 41566230 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_144349_41566230.yaml "
" HCC 1043950 54593731 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_144349_54593731.yaml "
" HCC 1043950 57660249 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_144349_57660249.yaml "
" HCC 1043950 72125941 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_144349_72125941.yaml "
" HCC 1043950 99434948 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_144349_99434948.yaml "
" HCC 1044300 63592626 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_144349_63592626.yaml "
" HCC 1044300 66834161 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_144349_66834161.yaml "
" HCC 1044300 92910319 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_144349_92910319.yaml "
" HCC 1044300 94613541 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_144349_94613541.yaml "
" HCC 1044300 97569363 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_144349_97569363.yaml "
" HCC 1044649 24805208 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_144349_24805208.yaml "
" HCC 1044649 46366417 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_144349_46366417.yaml "
" HCC 1044649 59396015 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_144349_59396015.yaml "
" HCC 1044649 84121621 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_144349_84121621.yaml "
" HCC 1044649 94560988 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_144349_94560988.yaml "
" MOPP 1052650 29079471 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_145913_29079471.yaml "
" MOPP 1052650 31571230 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_145913_31571230.yaml "
" MOPP 1052650 45373570 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_145913_45373570.yaml "
" MOPP 1052650 68839073 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_145913_68839073.yaml "
" MOPP 1052650 85763600 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_145913_85763600.yaml "
" MOPP 1053650 35281577 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_145913_35281577.yaml "
" MOPP 1053650 39888091 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_145913_39888091.yaml "
" MOPP 1053650 59550066 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_145913_59550066.yaml "
" MOPP 1053650 78038978 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_145913_78038978.yaml "
" MOPP 1053650 82093235 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_145913_82093235.yaml "
" MOPP 1054650 3611780 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_145913_03611780.yaml "
" MOPP 1054650 21247851 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_145913_21247851.yaml "
" MOPP 1054650 26628185 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_145913_26628185.yaml "
" MOPP 1054650 60567645 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_145913_60567645.yaml "
" MOPP 1054650 94967765 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_145913_94967765.yaml "
" MOPP 1055650 34097792 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_145913_34097792.yaml "
" MOPP 1055650 44866707 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_145913_44866707.yaml "
" MOPP 1055650 61810606 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_145913_61810606.yaml "
" MOPP 1055650 79924848 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_145913_79924848.yaml "
" MOPP 1055650 83145544 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_145913_83145544.yaml "
" MOPP 1056649 17666981 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_145913_17666981.yaml "
" MOPP 1056649 68347478 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_145913_68347478.yaml "
" MOPP 1056649 73504121 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_145913_73504121.yaml "
" MOPP 1056649 88486608 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_145913_88486608.yaml "
" MOPP 1056649 92808036 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_145913_92808036.yaml "
)
oth=(
" HC 1030000 15879716 results/netclamp/network_clamp.optimize.HC_1030000_20210211_151509_15879716.yaml "
" HC 1032250 39786206 results/netclamp/network_clamp.optimize.HC_1032250_20210211_151509_39786206.yaml "
" HC 1034500 17324444 results/netclamp/network_clamp.optimize.HC_1034500_20210211_151509_17324444.yaml "
" HC 1036750 96962730 results/netclamp/network_clamp.optimize.HC_1036750_20210211_151509_96962730.yaml "
" HC 1038999 69758687 results/netclamp/network_clamp.optimize.HC_1038999_20210211_151509_69758687.yaml "
" BC 1039000 93454042 results/netclamp/network_clamp.optimize.BC_1039000_20210211_151509_93454042.yaml "
" BC 1039950 27431042 results/netclamp/network_clamp.optimize.BC_1039950_20210211_151509_27431042.yaml "
" BC 1040900 32603857 results/netclamp/network_clamp.optimize.BC_1040900_20210211_151509_32603857.yaml "
" BC 1041850 87742037 results/netclamp/network_clamp.optimize.BC_1041850_20210211_151509_87742037.yaml "
" BC 1042799 183453 results/netclamp/network_clamp.optimize.BC_1042799_20210211_151509_00183453.yaml "
" AAC 1042800 4893658 results/netclamp/network_clamp.optimize.AAC_1042800_20210211_151509_04893658.yaml "
" AAC 1042913 291547 results/netclamp/network_clamp.optimize.AAC_1042913_20210211_151509_00291547.yaml "
" AAC 1043025 62387369 results/netclamp/network_clamp.optimize.AAC_1043025_20210211_151509_62387369.yaml "
" AAC 1043138 19281943 results/netclamp/network_clamp.optimize.AAC_1043138_20210211_151509_19281943.yaml "
" AAC 1043249 54652217 results/netclamp/network_clamp.optimize.AAC_1043249_20210211_151509_54652217.yaml "
)
tho=(
" IS 1049650 4259860 results/netclamp/network_clamp.optimize.IS_1049650_20210211_151509_04259860.yaml "
" IS 1050400 63796673 results/netclamp/network_clamp.optimize.IS_1050400_20210211_151509_63796673.yaml "
" IS 1051150 22725943 results/netclamp/network_clamp.optimize.IS_1051150_20210211_151509_22725943.yaml "
" IS 1051900 82185961 results/netclamp/network_clamp.optimize.IS_1051900_20210211_151509_82185961.yaml "
" IS 1052649 45707385 results/netclamp/network_clamp.optimize.IS_1052649_20210211_151509_45707385.yaml "
)
smlrng=(
" HCC 1043250 71407528 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_71407528.yaml "
" HCC 1043250 92055940 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_144349_92055940.yaml "
" HCC 1043250 12260638 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_203406_12260638.yaml "
" HCC 1043250 17609813 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_203406_17609813.yaml "
" HCC 1043250 71407528 results/netclamp/network_clamp.optimize.HCC_1043250_20210211_203406_71407528.yaml "
" HCC 1043600 17293770 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_203406_17293770.yaml "
" HCC 1043600 42402504 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_203406_42402504.yaml "
" HCC 1043600 60991105 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_203406_60991105.yaml "
" HCC 1043600 64290895 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_203406_64290895.yaml "
" HCC 1043600 89433777 results/netclamp/network_clamp.optimize.HCC_1043600_20210211_203406_89433777.yaml "
" HCC 1043950 41566230 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_203406_41566230.yaml "
" HCC 1043950 54593731 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_203406_54593731.yaml "
" HCC 1043950 57660249 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_203406_57660249.yaml "
" HCC 1043950 72125941 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_203406_72125941.yaml "
" HCC 1043950 99434948 results/netclamp/network_clamp.optimize.HCC_1043950_20210211_203406_99434948.yaml "
" HCC 1044300 63592626 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_203406_63592626.yaml "
" HCC 1044300 66834161 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_203406_66834161.yaml "
" HCC 1044300 92910319 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_203406_92910319.yaml "
" HCC 1044300 94613541 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_203406_94613541.yaml "
" HCC 1044300 97569363 results/netclamp/network_clamp.optimize.HCC_1044300_20210211_203406_97569363.yaml "
" HCC 1044649 24805208 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_203406_24805208.yaml "
" HCC 1044649 46366417 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_203406_46366417.yaml "
" HCC 1044649 59396015 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_203406_59396015.yaml "
" HCC 1044649 84121621 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_203406_84121621.yaml "
" HCC 1044649 94560988 results/netclamp/network_clamp.optimize.HCC_1044649_20210211_203406_94560988.yaml "
" NGFC 1044650 12740157 results/netclamp/network_clamp.optimize.NGFC_1044650_20210211_203406_12740157.yaml "
" NGFC 1044650 93872787 results/netclamp/network_clamp.optimize.NGFC_1044650_20210211_203406_93872787.yaml "
" NGFC 1044650 95844113 results/netclamp/network_clamp.optimize.NGFC_1044650_20210211_203406_95844113.yaml "
" NGFC 1044650 96772370 results/netclamp/network_clamp.optimize.NGFC_1044650_20210211_203406_96772370.yaml "
" NGFC 1044650 97895890 results/netclamp/network_clamp.optimize.NGFC_1044650_20210211_203406_97895890.yaml "
" NGFC 1045900 6112188 results/netclamp/network_clamp.optimize.NGFC_1045900_20210211_203406_06112188.yaml "
" NGFC 1045900 67428613 results/netclamp/network_clamp.optimize.NGFC_1045900_20210211_203406_67428613.yaml "
" NGFC 1045900 71039025 results/netclamp/network_clamp.optimize.NGFC_1045900_20210211_203406_71039025.yaml "
" NGFC 1045900 89814943 results/netclamp/network_clamp.optimize.NGFC_1045900_20210211_203406_89814943.yaml "
" NGFC 1045900 95436908 results/netclamp/network_clamp.optimize.NGFC_1045900_20210211_203406_95436908.yaml "
" NGFC 1047150 27400566 results/netclamp/network_clamp.optimize.NGFC_1047150_20210211_203406_27400566.yaml "
" NGFC 1047150 48744644 results/netclamp/network_clamp.optimize.NGFC_1047150_20210211_203406_48744644.yaml "
" NGFC 1047150 50965365 results/netclamp/network_clamp.optimize.NGFC_1047150_20210211_203406_50965365.yaml "
" NGFC 1047150 59071557 results/netclamp/network_clamp.optimize.NGFC_1047150_20210211_203406_59071557.yaml "
" NGFC 1047150 77901687 results/netclamp/network_clamp.optimize.NGFC_1047150_20210211_203406_77901687.yaml "
" NGFC 1048400 35297351 results/netclamp/network_clamp.optimize.NGFC_1048400_20210211_203406_35297351.yaml "
" NGFC 1048400 35472841 results/netclamp/network_clamp.optimize.NGFC_1048400_20210211_203406_35472841.yaml "
" NGFC 1048400 38650070 results/netclamp/network_clamp.optimize.NGFC_1048400_20210211_203406_38650070.yaml "
" NGFC 1048400 62046131 results/netclamp/network_clamp.optimize.NGFC_1048400_20210211_203406_62046131.yaml "
" NGFC 1048400 80347840 results/netclamp/network_clamp.optimize.NGFC_1048400_20210211_203406_80347840.yaml "
" NGFC 1049649 10249569 results/netclamp/network_clamp.optimize.NGFC_1049649_20210211_203406_10249569.yaml "
" NGFC 1049649 26628153 results/netclamp/network_clamp.optimize.NGFC_1049649_20210211_203406_26628153.yaml "
" NGFC 1049649 77179804 results/netclamp/network_clamp.optimize.NGFC_1049649_20210211_203406_77179804.yaml "
" NGFC 1049649 89481705 results/netclamp/network_clamp.optimize.NGFC_1049649_20210211_203406_89481705.yaml "
" NGFC 1049649 99082330 results/netclamp/network_clamp.optimize.NGFC_1049649_20210211_203406_99082330.yaml "
" MOPP 1052650 29079471 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_203406_29079471.yaml "
" MOPP 1052650 31571230 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_203406_31571230.yaml "
" MOPP 1052650 45373570 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_203406_45373570.yaml "
" MOPP 1052650 68839073 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_203406_68839073.yaml "
" MOPP 1052650 85763600 results/netclamp/network_clamp.optimize.MOPP_1052650_20210211_203406_85763600.yaml "
" MOPP 1053650 35281577 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_203406_35281577.yaml "
" MOPP 1053650 39888091 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_203406_39888091.yaml "
" MOPP 1053650 59550066 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_203406_59550066.yaml "
" MOPP 1053650 78038978 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_203406_78038978.yaml "
" MOPP 1053650 82093235 results/netclamp/network_clamp.optimize.MOPP_1053650_20210211_203406_82093235.yaml "
" MOPP 1054650 3611780 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_203406_03611780.yaml "
" MOPP 1054650 21247851 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_203406_21247851.yaml "
" MOPP 1054650 26628185 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_203406_26628185.yaml "
" MOPP 1054650 60567645 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_203406_60567645.yaml "
" MOPP 1054650 94967765 results/netclamp/network_clamp.optimize.MOPP_1054650_20210211_203406_94967765.yaml "
" MOPP 1055650 34097792 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_203406_34097792.yaml "
" MOPP 1055650 44866707 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_203406_44866707.yaml "
" MOPP 1055650 61810606 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_203406_61810606.yaml "
" MOPP 1055650 79924848 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_203406_79924848.yaml "
" MOPP 1055650 83145544 results/netclamp/network_clamp.optimize.MOPP_1055650_20210211_203406_83145544.yaml "
" MOPP 1056649 17666981 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_203406_17666981.yaml "
" MOPP 1056649 68347478 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_203406_68347478.yaml "
" MOPP 1056649 73504121 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_203406_73504121.yaml "
" MOPP 1056649 88486608 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_203406_88486608.yaml "
" MOPP 1056649 92808036 results/netclamp/network_clamp.optimize.MOPP_1056649_20210211_203406_92808036.yaml "
)

N_cores=1

IFS='
'
counter=0
for f in ${smlrng[@]}
do

set -- "$f" 
IFS=" " ; declare -a tempvar=($*) 


#ibrun -n 56 -o  0 task_affinity ./mycode.exe input1 &   # 56 tasks; offset by  0 entries in hostfile.
#ibrun -n 56 -o 56 task_affinity ./mycode.exe input2 &   # 56 tasks; offset by 56 entries in hostfile.
#wait                                                    # Required; else script will exit immediately.

#pop=${tempvar[0]:1:-1}
pop=${tempvar[0]}
gid=${tempvar[1]}
seed=${tempvar[2]}
yaml=${tempvar[3]}

ibrun -n $N_cores -o $counter python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
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
    --opt-seed $seed \
    --params-path $yaml &



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
