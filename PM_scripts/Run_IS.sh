export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=no
ml load intel19

fil=(
"'IS', 1049650, 'results/netclamp/network_clamp.optimize.IS_1049650_20210131_195001.yaml'"
"'IS', 1049650, 'results/netclamp/network_clamp.optimize.IS_1049650_20210201_133845.yaml'"
"'IS', 1049650, 'results/netclamp/network_clamp.optimize.IS_1049650_20210201_140528.yaml'"
"'IS', 1049650, 'results/netclamp/network_clamp.optimize.IS_1049650_20210201_143252.yaml'"
"'IS', 1049650, 'results/netclamp/network_clamp.optimize.IS_1049650_20210201_210607.yaml'"
"'IS', 1050400, 'results/netclamp/network_clamp.optimize.IS_1050400_20210131_190000.yaml'"
"'IS', 1050400, 'results/netclamp/network_clamp.optimize.IS_1050400_20210131_215442.yaml'"
"'IS', 1050400, 'results/netclamp/network_clamp.optimize.IS_1050400_20210131_234302.yaml'"
"'IS', 1050400, 'results/netclamp/network_clamp.optimize.IS_1050400_20210131_234611.yaml'"
"'IS', 1050400, 'results/netclamp/network_clamp.optimize.IS_1050400_20210201_213534.yaml'"
"'IS', 1051150, 'results/netclamp/network_clamp.optimize.IS_1051150_20210131_183315.yaml'"
"'IS', 1051150, 'results/netclamp/network_clamp.optimize.IS_1051150_20210131_213145.yaml'"
"'IS', 1051150, 'results/netclamp/network_clamp.optimize.IS_1051150_20210131_232145.yaml'"
"'IS', 1051150, 'results/netclamp/network_clamp.optimize.IS_1051150_20210131_232146.yaml'"
"'IS', 1051150, 'results/netclamp/network_clamp.optimize.IS_1051150_20210131_232444.yaml'"
"'IS', 1051900, 'results/netclamp/network_clamp.optimize.IS_1051900_20210131_192348.yaml'"
"'IS', 1051900, 'results/netclamp/network_clamp.optimize.IS_1051900_20210131_222503.yaml'"
"'IS', 1051900, 'results/netclamp/network_clamp.optimize.IS_1051900_20210201_000444.yaml'"
"'IS', 1051900, 'results/netclamp/network_clamp.optimize.IS_1051900_20210201_205350.yaml'"
"'IS', 1051900, 'results/netclamp/network_clamp.optimize.IS_1051900_20210202_011728.yaml'"
"'IS', 1052649, 'results/netclamp/network_clamp.optimize.IS_1052649_20210131_180930.yaml'"
"'IS', 1052649, 'results/netclamp/network_clamp.optimize.IS_1052649_20210131_210721.yaml'"
"'IS', 1052649, 'results/netclamp/network_clamp.optimize.IS_1052649_20210131_230028.yaml'"
"'IS', 1052649, 'results/netclamp/network_clamp.optimize.IS_1052649_20210131_230317.yaml'"
"'IS', 1052649, 'results/netclamp/network_clamp.optimize.IS_1052649_20210201_212719.yaml'"
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

