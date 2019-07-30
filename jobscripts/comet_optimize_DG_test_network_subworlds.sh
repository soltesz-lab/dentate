#!/bin/bash
#
#SBATCH -J optimize_DG_test_network_subworlds
#SBATCH -o ./results/optimize_DG_test_network_subworlds.%j.o
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=12
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5

set -x

export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=${MODEL_HOME}/dentate
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/optimize_DG_test_network_subworlds_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin


ibrun -np 576 python3 ./scripts/main.py \
    python3 -m nested.optimize  \
    --config-file-path=$DG_HOME/config/DG_test_network_subworlds_config.yaml \
    --output-dir=$results_path \
    --pop_size=4 \
    --max_iter=4 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --procs-per-worker=144 \
    --no-cleanup \
    --verbose \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --cell_selection_path=$DG_HOME/datasets/DG_slice_20190728.yaml \
    --spike_input_path="$SCRATCH/dentate/DG_input_spike_trains_20190729.h5" \
    --spike_input_namespace='Input Spikes' \
    --max-walltime-hours=3.75 \
    --io-size=64 \
    -v



