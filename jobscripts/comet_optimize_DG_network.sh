#!/bin/bash
#
#SBATCH -J optimize_DG_network
#SBATCH -o ./results/optimize_DG_network.%j.o
#SBATCH --nodes=60
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

#ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/optimize_DG_network_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

. $HOME/comet_env.sh

#ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 3 \
   `which python3` optimize_network.py \                                                                                       --config-file-path=$DG_HOME/config/DG_optimize_network.yaml \                                                           --output-dir=$results_path \                                                                                            --param-config-name="Weight inh microcircuit" \                                                                         --target-rate-map-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \                 --target-rate-map-namespace="Place Selectivity" \                                                                       --verbose \                                                                                                             --nprocs_per_worker=716 \                                                                                               --no_cleanup \                                                                                                          --arena_id=A --trajectory_id=Diag \                                                                                     --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \                                               --dataset_prefix="$SCRATCH/dentate" \                                                                                   --config_prefix=$DG_HOME/config \                                                                                       --results_path=$results_path \                                                                                          --spike_input_path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \                 --spike_input_namespace='Input Spikes A Diag' \                                                                         --spike_input_attr='Spike Train' \                                                                                      --max_walltime_hours=2.0 \                                                                                              --io_size=8 \                                                                                                           --microcircuit_inputs \                                                                                                 --use_coreneuron \                                                                                                      -v 


