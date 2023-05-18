#!/bin/bash

#SBATCH -J optimize_DG_network_neg2000_neg1900um # Job name
#SBATCH -o ./results/optimize_DG_network_neg2000_neg1900um_test.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_network_neg2000_neg1900um_test.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 4             # Total # of nodes 
#SBATCH --ntasks-per-node=56 # # of mpi tasks per node
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load mkl/19.1.1
module load python3/3.9.2
module load phdf5/1.10.4

set -x

export TF_CPP_MIN_LOG_LEVEL=2

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH

export DATA_PREFIX="/tmp/optimize_DG_network"
export CDTools=/home1/apps/CDTools/1.2

export PATH=${CDTools}/bin:$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=4

export UCX_TLS="knem,dc_x"

results_path=$SCRATCH/dentate/results/optimize_DG_network
export results_path

mkdir -p $results_path

cd $SLURM_SUBMIT_DIR

distribute.bash ${SCRATCH}/dentate/optimize_DG_network

ibrun -np 224 \
    python3 optimize_network.py \
    --config-path=$DG_HOME/config/DG_optimize_network_neg2000_neg1900um_test.yaml \
    --optimize-file-dir=$results_path \
    --optimize-file-name=dmosopt.optimize_network_20230308_1600.h5 \
    --target-features-path="$DATA_PREFIX/Full_Scale_Control/DG_input_features_20220216.h5" \
    --target-features-namespace="Place Selectivity" \
    --verbose \
    --nprocs-per-worker=223 \
    --n-epochs=1 \
    --n-initial=50 --initial-method="slh" --num-generations=400 --population-size=200 --resample-fraction 1.0 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$DG_HOME/templates \
    --dataset_prefix="$DATA_PREFIX" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$DATA_PREFIX/Slice/dentatenet_Slice_SLN_neg2000_neg1900um_20221210_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=24.0 \
    --io_size=1 \
    --use_cell_attr_gen \
    --microcircuit_inputs \
    --verbose

