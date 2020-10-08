#!/bin/bash
#
#SBATCH -J dentate_sample_extent
#SBATCH -o ./results/dentate_sample_extent.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p shared
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

set -x

results_path=$SCRATCH/dentate/results/DG_sample_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

ibrun -np 1 -v python3 ./scripts/sample_extent.py \
    --arena-id='A' \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/dentate" \
    --output-path=$results_path \
    --spike-input-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20200323_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --input-features-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200611_compressed.h5" \
    --output-path=${results_path} \
    --bin-sample-count=10 \
    --distance-bin-extent=500. \
    --write-selection \
    -i GC -i MC \
    --verbose

