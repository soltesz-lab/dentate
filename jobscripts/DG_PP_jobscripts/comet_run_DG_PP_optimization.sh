#!/bin/bash
#
#SBATCH -J optimize_DG_PP_features
#SBATCH -n 200
#SBATCH -t 4:00:00
#SBATCH -o ./results/optimize_DG_PP_features.%j.o
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/scripts:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

CONFIG_DIR=$HOME/model/dentate/config

set -x

results_path=$SCRATCH/dentate/results/DG_PP_optimize_$SLURM_JOB_ID
mkdir -p $results_path

ibrun -np 200 python -m nested.optimize \
    --config-file-path=$CONFIG_DIR/DG-PP/optimize_DG_PP_config_place_module_${SLURM_ARRAY_TASK_ID}.yaml \
    --pop-size=200         \
    --max-iter=50          \
    --path-length=3        \
    --disp                 \
    --output-dir="$results_path"  \
    --label=${SLURM_ARRAY_TASK_ID} \
    --input_params_file_path=$CONFIG_DIR/Input_Features.yaml


