#!/bin/bash

#SBATCH -J optimize_DG_test_network_test # Job name
#SBATCH -o ./results/optimize_DG_network_test.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_network_test.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 32             # Total # of nodes 
#SBATCH --ntasks-per-node=56 # # of mpi tasks per node
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export FI_MLX_ENABLE_SPAWN=1

#export I_MPI_ADJUST_ALLREDUCE=6
#export I_MPI_ADJUST_SCATTERV=2
#export I_MPI_ADJUST_ALLGATHER=4
#export I_MPI_ADJUST_ALLGATHERV=4
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2

results_path=$SCRATCH/dentate/results/optimize_DG_network_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

cd $SLURM_SUBMIT_DIR


mpirun -n 4 python3 optimize_network.py \
    --config-path=$DG_HOME/config/DG_optimize_network_test.yaml \
    --output-dir=$results_path \
    --target-rate-map-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --target-rate-map-namespace="Place Selectivity" \
    --verbose \
    --nprocs-per-worker=440 \
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
    -v



