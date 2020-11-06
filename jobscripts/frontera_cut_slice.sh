#!/bin/bash

#SBATCH -J dentate_cut_slice # Job name
#SBATCH -o ./results/dentate_cut_slice.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate_cut_slice.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 30             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load intel/18.0.5
module load python3
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

results_path=$SCRATCH/dentate/results/cut_slice_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

export I_MPI_EXTRA_FILESYSTEM=enable
export I_MPI_EXTRA_FILESYSTEM_LIST=lustre
#export I_MPI_ADJUST_ALLGATHER=4
#export I_MPI_ADJUST_ALLGATHERV=4
#export I_MPI_ADJUST_ALLTOALL=4

export PYTHON=`which python3`

ibrun python3 ./scripts/cut_slice.py \
    --arena-id=A --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/striped/dentate" \
    --output-path=$results_path \
    --io-size=96 \
    --spike-input-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h\
5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --distance-limits -150 150 \
    --write-selection \
    --verbose
