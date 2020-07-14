#!/bin/bash
#
#SBATCH -J optimize_DG_network_clamp_pf
#SBATCH -o ./results/optimize_DG_network_clamp_pf.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH -p compute
#SBATCH -t 6:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

. $HOME/comet_env.sh

#ulimit -c unlimited
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export MPLBACKEND=SVG

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`
results_path=$SCRATCH/dentate/results/optimize_DG_network_clamp_pf_$SLURM_JOB_ID
export results_path
mkdir -p ${results_path}

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 96 \
 python3 -m nested.optimize \
    --storage-file-path $SCRATCH/dentate/results/optimize_DG_network_clamp_pf_34662854/20200713_211315_DG_optimize_network_clamp_PopulationAnnealing_optimization_history.hdf5 \
    --config-file-path=$DG_HOME/config/DG_optimize_network_clamp_config.yaml \
    --output-dir=${results_path} \
    --pop_size=95 \
    --max_iter=2000 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=1 \
    --no_cleanup \
    --gid=126002 \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$SCRATCH/dentate/results/netclamp \
    --input_features_path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200611_compressed.h5" \
    --param_config_name="Weight all inh soma dend" \
    --arena_id=A --trajectory_id=Diag \
    --n_trials=1 \
    --target_rate_map_path=$SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20200327.h5

