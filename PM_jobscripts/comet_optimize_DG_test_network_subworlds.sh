#!/bin/bash
#
#SBATCH -J optimize_DG_network_subworlds
#SBATCH -o ./results/optimize_DG_network_subworlds.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

set -x

export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/optimize_DG_test_network_subworlds_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun python3 -m nested.optimize \
    --config-file-path=$DG_HOME/config/DG_optimize_network_subworlds_config_dbg.yaml \
    --output-dir=$results_path \
    --pop_size=1 \
    --max_iter=1 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=48 \
    --io_size=8 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike_input_namespace='Input Spikes' \
    --max_walltime_hours=1.9 \
    -v



