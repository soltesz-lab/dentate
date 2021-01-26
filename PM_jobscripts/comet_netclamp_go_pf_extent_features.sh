#!/bin/bash
#
#SBATCH -J netclamp_go_pf_extent_features
#SBATCH -o ./results/netclamp_go_pf_extent_features.%j.o
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#
. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 24 \
`which python3` network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_S_extent.yaml -p GC -t 14000 \
    --template-paths templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path "$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200321_compressed.h5" \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id DDiag  --n-trials 5 \
    --config-prefix config

