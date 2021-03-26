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
" HCC 1043250 33236209 "
" HCC 1043250 92055940 "
" HCC 1043250 71407528 "
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
    -p HCC -g 1043250 -g 1043600 -g 1043950 -g 1044300 -g 1044649 -t 9500 --dt 0.001 \
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
