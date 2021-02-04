export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
ml load intel19

fil=(
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210131_195001.yaml'"
"'AAC', 1042913, 'results/netclamp/network_clamp.optimize.AAC_1042913_20210131_190000.yaml'"
"'AAC', 1042913, 'results/netclamp/network_clamp.optimize.AAC_1042913_20210131_215442.yaml'"
"'AAC', 1042913, 'results/netclamp/network_clamp.optimize.AAC_1042913_20210131_234302.yaml'"
"'AAC', 1042913, 'results/netclamp/network_clamp.optimize.AAC_1042913_20210131_234611.yaml'"
"'AAC', 1042913, 'results/netclamp/network_clamp.optimize.AAC_1042913_20210201_213534.yaml'"
"'AAC', 1043025, 'results/netclamp/network_clamp.optimize.AAC_1043025_20210131_183315.yaml'"
"'AAC', 1043025, 'results/netclamp/network_clamp.optimize.AAC_1043025_20210131_213145.yaml'"
"'AAC', 1043025, 'results/netclamp/network_clamp.optimize.AAC_1043025_20210131_232145.yaml'"
"'AAC', 1043025, 'results/netclamp/network_clamp.optimize.AAC_1043025_20210131_232146.yaml'"
"'AAC', 1043025, 'results/netclamp/network_clamp.optimize.AAC_1043025_20210131_232444.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210131_192348.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210131_222503.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210201_000444.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210201_205350.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210202_004417.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_180930.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_210721.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_230028.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_230317.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210202_004417.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_133845.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_140528.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_143252.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_210607.yaml'"
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

sleep 1m

done

