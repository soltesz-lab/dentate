#!/bin/bash
#
#SBATCH -J optimize_DG_network
#SBATCH -o ./results/optimize_DG_network.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 7:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project

#results_path=$SCRATCH/dentate/results/optimize_DG_network_$SLURM_JOB_ID
results_path=/oasis/scratch/comet/iraikov/temp_project/dentate/results/optimize_DG_network_37370866
export results_path

#mkdir -p $results_path

ibrun -n 9 python3 optimize_network.py \
    --config-path=$DG_HOME/config/DG_optimize_network.yaml \
    --results-dir=$results_path \
    --target-rate-map-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --target-rate-map-namespace="Place Selectivity" \
    --verbose \
    --nprocs-per-worker=190 \
    --n-iter=1 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=7 \
    --io_size=24 \
    --microcircuit_inputs \
    --verbose



