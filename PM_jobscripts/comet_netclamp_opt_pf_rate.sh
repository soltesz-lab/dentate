#!/bin/bash
#
#SBATCH -J netclamp_opt_pf_extent_features
#SBATCH -o ./results/netclamp_opt_pf_extent_features.%j.o
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#
. $HOME/comet_env.sh

#ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 96 \
`which python3` network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC -g $1 --n-trials 3 -t 9500 \
    --template-paths templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path "$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200611_compressed.h5" \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 6000 --opt-epsilon 1 \
    --param-config-name 'Weight all no MC inh soma all-dend' \
    --arena-id A --trajectory-id Diag \
    --target-features-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20200901.h5 \
    --use-coreneuron \
    selectivity_rate
