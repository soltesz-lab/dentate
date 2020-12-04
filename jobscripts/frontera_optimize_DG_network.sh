#!/bin/bash

#SBATCH -J optimize_DG_test_network # Job name
#SBATCH -o ./results/optimize_DG_network.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_network.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 160             # Total # of nodes 
#SBATCH --ntasks-per-node=56 # # of mpi tasks per node
#SBATCH -t 6:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load intel/18.0.5
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3/intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export FI_MLX_ENABLE_SPAWN=1

export I_MPI_SCALABLE_OPTIMIZATION=1
export I_MPI_ADJUST_ALLREDUCE=6
export I_MPI_ADJUST_SCATTERV=2

results_path=$SCRATCH/dentate/results/optimize_DG_network_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

cd $SLURM_SUBMIT_DIR

export hosts=`scontrol show hostname $SLURM_NODELIST | paste -s -d, -`

MY_MPIRUN_OPTIONS="-rr -hosts $hosts" ibrun -n 11 python3 optimize_network.py \
    --config-path=$DG_HOME/config/DG_optimize_network.yaml \
    --output-dir=$results_path \
    --target-rate-map-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --target-rate-map-namespace="Place Selectivity" \
    --verbose \
    --nprocs-per-worker=894 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/striped/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=2.0 \
    --io_size=8 \
    --microcircuit_inputs \
    --use_coreneuron \
    --verbose



