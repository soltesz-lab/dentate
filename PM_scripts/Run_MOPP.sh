export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=no
ml load intel19

fil=(
"'MOPP', 1052650, 'results/netclamp/network_clamp.optimize.MOPP_1052650_20210131_195001.yaml'"
"'MOPP', 1052650, 'results/netclamp/network_clamp.optimize.MOPP_1052650_20210201_133845.yaml'"
"'MOPP', 1052650, 'results/netclamp/network_clamp.optimize.MOPP_1052650_20210201_140528.yaml'"
"'MOPP', 1052650, 'results/netclamp/network_clamp.optimize.MOPP_1052650_20210201_143252.yaml'"
"'MOPP', 1052650, 'results/netclamp/network_clamp.optimize.MOPP_1052650_20210201_210607.yaml'"
"'MOPP', 1053650, 'results/netclamp/network_clamp.optimize.MOPP_1053650_20210131_185959.yaml'"
"'MOPP', 1053650, 'results/netclamp/network_clamp.optimize.MOPP_1053650_20210131_215442.yaml'"
"'MOPP', 1053650, 'results/netclamp/network_clamp.optimize.MOPP_1053650_20210131_234302.yaml'"
"'MOPP', 1053650, 'results/netclamp/network_clamp.optimize.MOPP_1053650_20210131_234611.yaml'"
"'MOPP', 1053650, 'results/netclamp/network_clamp.optimize.MOPP_1053650_20210201_213534.yaml'"
"'MOPP', 1054650, 'results/netclamp/network_clamp.optimize.MOPP_1054650_20210131_183315.yaml'"
"'MOPP', 1054650, 'results/netclamp/network_clamp.optimize.MOPP_1054650_20210131_213145.yaml'"
"'MOPP', 1054650, 'results/netclamp/network_clamp.optimize.MOPP_1054650_20210131_232145.yaml'"
"'MOPP', 1054650, 'results/netclamp/network_clamp.optimize.MOPP_1054650_20210131_232146.yaml'"
"'MOPP', 1054650, 'results/netclamp/network_clamp.optimize.MOPP_1054650_20210131_232444.yaml'"
"'MOPP', 1055650, 'results/netclamp/network_clamp.optimize.MOPP_1055650_20210131_192348.yaml'"
"'MOPP', 1055650, 'results/netclamp/network_clamp.optimize.MOPP_1055650_20210131_222503.yaml'"
"'MOPP', 1055650, 'results/netclamp/network_clamp.optimize.MOPP_1055650_20210201_000444.yaml'"
"'MOPP', 1055650, 'results/netclamp/network_clamp.optimize.MOPP_1055650_20210201_205350.yaml'"
"'MOPP', 1055650, 'results/netclamp/network_clamp.optimize.MOPP_1055650_20210202_011728.yaml'"
"'MOPP', 1056649, 'results/netclamp/network_clamp.optimize.MOPP_1056649_20210131_180930.yaml'"
"'MOPP', 1056649, 'results/netclamp/network_clamp.optimize.MOPP_1056649_20210131_210721.yaml'"
"'MOPP', 1056649, 'results/netclamp/network_clamp.optimize.MOPP_1056649_20210131_230028.yaml'"
"'MOPP', 1056649, 'results/netclamp/network_clamp.optimize.MOPP_1056649_20210131_230317.yaml'"
"'MOPP', 1056649, 'results/netclamp/network_clamp.optimize.MOPP_1056649_20210201_212719.yaml'"
)
IFS='
'
for f in ${fil[@]}
do

set -- "$f" 
IFS=", "; declare -a tempvar=($*) 
echo "${tempvar[0]}", "${tempvar[2]}"

python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p ${tempvar[0]:1:-1} -g ${tempvar[1]} -t 9500 --dt 0.001 \
    --dataset-prefix /scratch1/03320/iraikov/striped/dentate \
    --config-prefix config \
    --input-features-path /scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp \
    --params-path ${tempvar[2]:1:-1} 

done

