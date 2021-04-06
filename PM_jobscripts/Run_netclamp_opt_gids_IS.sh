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

fil=(
" IS 1049650 4259860 "
" IS 1049650 11745958 "
" IS 1049650 75940072 "
" IS 1049650 49627038 "
" IS 1049650 84013649 "
" IS 1050400 63796673 "
" IS 1050400 69320701 "
" IS 1050400 7843435 "
" IS 1050400 10084233 "
" IS 1050400 93591428 "
" IS 1051150 22725943 "
" IS 1051150 21032749 "
" IS 1051150 1339500 "
" IS 1051150 83916441 "
" IS 1051150 49587749 "
" IS 1051900 82185961 "
" IS 1051900 27654574 "
" IS 1051900 23672271 "
" IS 1051900 70119958 "
" IS 1051900 51871840 "
" IS 1052649 45707385 "
" IS 1052649 37549278 "
" IS 1052649 18680556 "
" IS 1052649 60814941 "
" IS 1052649 82004212 "
)

N_cores=35

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
ibrun -n $N_cores -o $((counter * 56))  python3  network_clamp.py optimize -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    --template-paths templates \
    -p IS -g 1049650 -g 1050400 -g 1051150 -g 1051900 -g 1052649 -t 9500 --dt 0.001 \
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
    --n-trials 4 \
    --opt-iter 400 rate & 


#    --results-file network_clamp.optimize.$pop\_$gid\_$(date +'%Y%m%d_%H%M%S')\_$seed.h5 \

counter=$((counter + 1))

done
wait
