fil=(
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210131_195001.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_133845.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_140528.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_143252.yaml'"
"'AAC', 1042800, 'results/netclamp/network_clamp.optimize.AAC_1042800_20210201_210607.yaml'"
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
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210201_444.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210201_205350.yaml'"
"'AAC', 1043138, 'results/netclamp/network_clamp.optimize.AAC_1043138_20210202_4417.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_180930.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_210721.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_230028.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210131_230317.yaml'"
"'AAC', 1043249, 'results/netclamp/network_clamp.optimize.AAC_1043249_20210202_4417.yaml'"
)
IFS='
'
for f in ${fil[@]}
do

set -- "$f" 
IFS=", "; declare -a tempvar=($*) 
echo "${tempvar[0]}", "${tempvar[2]}"
done
