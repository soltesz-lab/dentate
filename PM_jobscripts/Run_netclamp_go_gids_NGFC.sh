#!/bin/bash
#
#SBATCH -J netclamp_single_cell 
#SBATCH -o ./results/netclamp_single_cell.%j.o
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=56
#SBATCH -p normal
#SBATCH -t 4:00:00
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=END
#


export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

testset=(
" NGFC 1044650 1045900 1047150 1048400 1049640 12740157 "
)
settest=(
" HC 1030000 1032250 1034500 1036750 1038999 15879716 "
" MOPP 1052650 1053650 1054650 1055650 1056649 31571230 "
" BC 1039000 1039950 1040900 1041850 1042799 93454042 "
" HCC 1043250 1043600 1043950 1044300 1044649 33236209 "
" AAC 1042800 1042913 1043025 1043138 1043249 4893658 "
" IS 1049650 1050400 1051150 1051900 1052649 4259860 "
" NGFC 1044650 1045900 1047150 1048400 1049640 12740157 "
)
fil=(
" AAC 1042800 1042913 1043025 1043138 1043249 4893658 "
" AAC 1042800 1042913 1043025 1043138 1043249 27137089 "
" AAC 1042800 1042913 1043025 1043138 1043249 36010476 "
" AAC 1042800 1042913 1043025 1043138 1043249 53499406 "
" AAC 1042800 1042913 1043025 1043138 1043249 49937004 "
" BC 1039000 1039950 1040900 1041850 1042799 93454042 "
" BC 1039000 1039950 1040900 1041850 1042799 74865768 "
" BC 1039000 1039950 1040900 1041850 1042799 1503844 "
" BC 1039000 1039950 1040900 1041850 1042799 52357252 "
" BC 1039000 1039950 1040900 1041850 1042799 28135771 "
" HCC 1043250 1043600 1043950 1044300 1044649 33236209 "
" HCC 1043250 1043600 1043950 1044300 1044649 92055940 "
" HCC 1043250 1043600 1043950 1044300 1044649 71407528 "
" HCC 1043250 1043600 1043950 1044300 1044649 17609813 "
" HCC 1043250 1043600 1043950 1044300 1044649 12260638 "
" HC 1030000 1032250 1034500 1036750 1038999 15879716 "
" HC 1030000 1032250 1034500 1036750 1038999 45419272 "
" HC 1030000 1032250 1034500 1036750 1038999 28682721 "
" HC 1030000 1032250 1034500 1036750 1038999 53736785 "
" HC 1030000 1032250 1034500 1036750 1038999 63599789 "
" IS 1049650 1050400 1051150 1051900 1052649 4259860 "
" IS 1049650 1050400 1051150 1051900 1052649 11745958 "
" IS 1049650 1050400 1051150 1051900 1052649 75940072 "
" IS 1049650 1050400 1051150 1051900 1052649 49627038 "
" IS 1049650 1050400 1051150 1051900 1052649 84013649 "
" MOPP 1052650 1053650 1054650 1055650 1056649 31571230 "
" MOPP 1052650 1053650 1054650 1055650 1056649 45373570 "
" MOPP 1052650 1053650 1054650 1055650 1056649 85763600 "
" MOPP 1052650 1053650 1054650 1055650 1056649 68839073 "
" MOPP 1052650 1053650 1054650 1055650 1056649 29079471 "
" NGFC 1044650 1045900 1047150 1048400 1049640 12740157 "
" NGFC 1044650 1045900 1047150 1048400 1049640 97895890 "
" NGFC 1044650 1045900 1047150 1048400 1049640 93872787 "
" NGFC 1044650 1045900 1047150 1048400 1049640 95844113 "
" NGFC 1044650 1045900 1047150 1048400 1049640 96772370 "
)


IFS='
'
N_cores_gid=6
counter=0

#for f in ${fil[@]}
for f in ${testset[@]}
    do
    
    set -- "$f" 
    IFS=" " ; declare -a tempvar=($*) 
    
    N_pars=${#tempvar[@]}
    N_gids=$((N_pars-2))
    pop=${tempvar[0]}
    gids=${tempvar[@]:1:N_gids}
    seed=${tempvar[-1]}
    N_cores=$((N_gids * N_cores_gid + 1 ))
    
    gid_cmd=""
    for gid in ${gids[@]}
        do
        gid_cmd+="-g ${gid} "
        done
    
    ibrun -n 56 -o $((counter)) python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
        --template-paths templates \
        -p $pop \
        $gid_cmd \
        -t 9500 --dt 0.001 \
        --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
        --config-prefix config \
        --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
        --input-features-namespaces 'Place Selectivity' \
        --input-features-namespaces 'Grid Selectivity' \
        --input-features-namespaces 'Constant Selectivity' \
        --arena-id A --trajectory-id Diag \
        --results-path results/netclamp \
        --n-trials 4 \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformTrialMedian.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformTrialMean.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformBestMode.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformBestMedian.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformBestMean.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificTrialMedian.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificTrialMean.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificBestMedian.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificBestMean.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_UniformTrialMode.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificTrialMode.yaml \
        --params-path /scratch1/04119/pmoolcha/HDM/dentate/config/SingleCellParamYamls/Interneurons/NGFC_20210325_225723_Combined_SpecificBestMode.yaml \
    
    counter=$((counter + 1))
    
    done
wait
#        --opt-iter 800 rate & 
#        --opt-seed $seed \



# --params-path results/netclamp/network_clamp.optimize.AAC_20210317_032406_27137089.yaml \
# --params-path results/netclamp/network_clamp.optimize.AAC_20210317_032406_53499406.yaml \
# --params-path results/netclamp/network_clamp.optimize.AAC_20210317_032407_04893658.yaml \
# --params-path results/netclamp/network_clamp.optimize.AAC_20210317_032407_36010476.yaml \
# --params-path results/netclamp/network_clamp.optimize.AAC_20210317_032407_49937004.yaml \
# --params-path results/netclamp/network_clamp.optimize.BC_20210317_032406_01503844.yaml \
# --params-path results/netclamp/network_clamp.optimize.BC_20210317_032406_93454042.yaml \
# --params-path results/netclamp/network_clamp.optimize.BC_20210317_032407_28135771.yaml \
# --params-path results/netclamp/network_clamp.optimize.BC_20210317_032407_52357252.yaml \
# --params-path results/netclamp/network_clamp.optimize.BC_20210317_032407_74865768.yaml \
# --params-path results/netclamp/network_clamp.optimize.HC_20210317_032407_15879716.yaml \
# --params-path results/netclamp/network_clamp.optimize.HC_20210317_032407_28682721.yaml \
# --params-path results/netclamp/network_clamp.optimize.HC_20210317_032407_45419272.yaml \
# --params-path results/netclamp/network_clamp.optimize.HC_20210317_032407_53736785.yaml \
# --params-path results/netclamp/network_clamp.optimize.HC_20210317_032407_63599789.yaml \
# --params-path results/netclamp/network_clamp.optimize.HCC_20210317_032406_12260638.yaml \
# --params-path results/netclamp/network_clamp.optimize.HCC_20210317_032407_33236209.yaml \
# --params-path results/netclamp/network_clamp.optimize.HCC_20210317_032407_71407528.yaml \
# --params-path results/netclamp/network_clamp.optimize.HCC_20210317_032407_92055940.yaml \
# --params-path results/netclamp/network_clamp.optimize.IS_20210317_032406_04259860.yaml \
# --params-path results/netclamp/network_clamp.optimize.IS_20210317_032406_11745958.yaml \
# --params-path results/netclamp/network_clamp.optimize.IS_20210317_032407_49627038.yaml \
# --params-path results/netclamp/network_clamp.optimize.IS_20210317_032407_75940072.yaml \
# --params-path results/netclamp/network_clamp.optimize.IS_20210317_032407_84013649.yaml \
# --params-path results/netclamp/network_clamp.optimize.MOPP_20210317_032406_85763600.yaml \
# --params-path results/netclamp/network_clamp.optimize.MOPP_20210317_032407_29079471.yaml \
# --params-path results/netclamp/network_clamp.optimize.MOPP_20210317_032407_31571230.yaml \
# --params-path results/netclamp/network_clamp.optimize.MOPP_20210317_032407_45373570.yaml \
# --params-path results/netclamp/network_clamp.optimize.MOPP_20210317_032407_68839073.yaml \
# --params-path results/netclamp/network_clamp.optimize.NGFC_20210317_032406_12740157.yaml \
# --params-path results/netclamp/network_clamp.optimize.NGFC_20210317_032406_93872787.yaml \
# --params-path results/netclamp/network_clamp.optimize.NGFC_20210317_032407_95844113.yaml \
# --params-path results/netclamp/network_clamp.optimize.NGFC_20210317_032407_96772370.yaml \
# --params-path results/netclamp/network_clamp.optimize.NGFC_20210317_032407_97895890.yaml \
