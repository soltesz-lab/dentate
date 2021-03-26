#!/bin/bash
#
#SBATCH -J netclamp_single_cell 
#SBATCH -o ./results/netclamp_single_cell.%j.o
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=56
#SBATCH -p normal
#SBATCH -t 5:00:00
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=END
#


export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=yes
module load python3
module load phdf5
ml load intel19

set -x

export MODEL_HOME=/scratch1/04119/pmoolcha/HDM
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export RAIKOVSCRATCH=/scratch1/03320/iraikov


fil=(
" MC 1000018 1007283 1016576 1024018 1029806  4893658 "
" MC 1000018 1007283 1016576 1024018 1029806 51265473 "
" MC 1000018 1007283 1016576 1024018 1029806 27902589 "
" MC 1000018 1007283 1016576 1024018 1029806 33624479 "
" MC 1000018 1007283 1016576 1024018 1029806 15388757 "
)


IFS='
'
N_cores_gid=30
counter=0

for f in ${fil[@]}
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
    
    ibrun -n $N_cores -o $((counter * 5 * 56)) -rr python3 network_clamp.py optimize  -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
        -p MC -g 1000018 -g 1007283 -g 1016576 -g 1024018 -g 1029806 \
        -t 9500 --dt 0.001 \
        --n-trials 4 --trial-regime best \
        --template-paths $DG_HOME/templates:$MODEL_HOME/dgc/Mateos-Aparicio2014 \
        --dataset-prefix $RAIKOVSCRATCH/striped/dentate \
        --results-path results/netclamp \
        --input-features-path $RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
        --input-features-namespaces 'Place Selectivity' \
        --input-features-namespaces 'Grid Selectivity' \
        --input-features-namespaces 'Constant Selectivity' \
        --config-prefix config  --opt-iter 4000  --opt-epsilon 1 \
        --param-config-name 'Weight exc inh netclamp' \
        --arena-id A --trajectory-id Diag \
        --target-features-path $RAIKOVSCRATCH/dentate/Slice/MC_extent_input_spike_trains_20201001.h5 \
        --opt-seed $seed \
        rate &
    
    counter=$((counter + 1))
    
    done
wait
