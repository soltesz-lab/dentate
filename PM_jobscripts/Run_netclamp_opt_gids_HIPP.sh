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
" HC 1030000 15879716 "
" HC 1030000 45419272 "
" HC 1030000 28682721 "
" HC 1030000 53736785 "
" HC 1030000 63599789 "
" HC 1032250 39786206 "
" HC 1032250 29585629 "
" HC 1032250 80350627 "
" HC 1032250 13571683 "
" HC 1032250 204284 "
" HC 1034500 17324444 "
" HC 1034500 19663111 "
" HC 1034500 48663281 "
" HC 1034500 85653587 "
" HC 1034500 61645166 "
" HC 1036750 96962730 "
" HC 1036750 96494881 "
" HC 1036750 9389064 "
" HC 1036750 72215563 "
" HC 1036750 86292756 "
" HC 1038999 69758687 "
" HC 1038999 19285510 "
" HC 1038999 57153876 "
" HC 1038999 45197076 "
" HC 1038999 58867820 "
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
    -p HC -g 1030000 -g 1032250 -g 1034500 -g 1036750 -g 1038999 -t 9500 --dt 0.001 \
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
