#!/bin/bash
#
#SBATCH -J dentate_sample_extent
#SBATCH -o ./results/dentate_cut_slice.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -p shared
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`
echo python is `which python3`

results_path=$SCRATCH/dentate/results/DG_sample_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 1536 \
`which python3` ./scripts/sample_extent.py \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/dentate" \
    --output-path=$results_path \
    --spike-input-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20190912_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --output-path=${results_path} \
    --bin-sample-count=1 \
    --write-selection \
    --verbose

