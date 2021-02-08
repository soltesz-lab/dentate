export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=no
ml load intel19

fil=(
"'BC', 1039000, 'results/netclamp/network_clamp.optimize.BC_1039000_20210131_195001.yaml'"
"'BC', 1039000, 'results/netclamp/network_clamp.optimize.BC_1039000_20210201_133845.yaml'"
"'BC', 1039000, 'results/netclamp/network_clamp.optimize.BC_1039000_20210201_140528.yaml'"
"'BC', 1039000, 'results/netclamp/network_clamp.optimize.BC_1039000_20210201_143252.yaml'"
"'BC', 1039000, 'results/netclamp/network_clamp.optimize.BC_1039000_20210201_210607.yaml'"
"'BC', 1039950, 'results/netclamp/network_clamp.optimize.BC_1039950_20210131_185959.yaml'"
"'BC', 1039950, 'results/netclamp/network_clamp.optimize.BC_1039950_20210131_215442.yaml'"
"'BC', 1039950, 'results/netclamp/network_clamp.optimize.BC_1039950_20210131_234302.yaml'"
"'BC', 1039950, 'results/netclamp/network_clamp.optimize.BC_1039950_20210131_234611.yaml'"
"'BC', 1039950, 'results/netclamp/network_clamp.optimize.BC_1039950_20210201_213534.yaml'"
"'BC', 1040900, 'results/netclamp/network_clamp.optimize.BC_1040900_20210131_183315.yaml'"
"'BC', 1040900, 'results/netclamp/network_clamp.optimize.BC_1040900_20210131_213145.yaml'"
"'BC', 1040900, 'results/netclamp/network_clamp.optimize.BC_1040900_20210131_232145.yaml'"
"'BC', 1040900, 'results/netclamp/network_clamp.optimize.BC_1040900_20210131_232146.yaml'"
"'BC', 1040900, 'results/netclamp/network_clamp.optimize.BC_1040900_20210131_232444.yaml'"
"'BC', 1041850, 'results/netclamp/network_clamp.optimize.BC_1041850_20210131_192348.yaml'"
"'BC', 1041850, 'results/netclamp/network_clamp.optimize.BC_1041850_20210131_222503.yaml'"
"'BC', 1041850, 'results/netclamp/network_clamp.optimize.BC_1041850_20210201_205350.yaml'"
"'BC', 1041850, 'results/netclamp/network_clamp.optimize.BC_1041850_20210202_004417.yaml'"
"'BC', 1041850, 'results/netclamp/network_clamp.optimize.BC_1041850_20210202_011728.yaml'"
"'BC', 1042799, 'results/netclamp/network_clamp.optimize.BC_1042799_20210131_180930.yaml'"
"'BC', 1042799, 'results/netclamp/network_clamp.optimize.BC_1042799_20210131_210721.yaml'"
"'BC', 1042799, 'results/netclamp/network_clamp.optimize.BC_1042799_20210131_230028.yaml'"
"'BC', 1042799, 'results/netclamp/network_clamp.optimize.BC_1042799_20210131_230317.yaml'"
"'BC', 1042799, 'results/netclamp/network_clamp.optimize.BC_1042799_20210201_212719.yaml'"
)
IFS='
'
counter=0
for f in ${fil[@]}
do

set -- "$f" 
IFS=", "; declare -a tempvar=($*) 
echo "${tempvar[0]}", "${tempvar[2]}"


#ibrun -n 56 -o  0 task_affinity ./mycode.exe input1 &   # 56 tasks; offset by  0 entries in hostfile.
#ibrun -n 56 -o 56 task_affinity ./mycode.exe input2 &   # 56 tasks; offset by 56 entries in hostfile.
#wait                                                    # Required; else script will exit immediately.


ibrun -n 1 -o $counter python3 network_clamp.py go -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
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
    --params-path ${tempvar[2]:1:-1} &

counter=$((counter+1))

done
wait
