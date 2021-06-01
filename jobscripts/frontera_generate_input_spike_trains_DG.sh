#!/bin/bash
#SBATCH -J generate_input_spike_trains_DG
#SBATCH -o ./results/generate_input_spike_trains_DG.%j.o
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 4             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5/1.10.4

set -x

export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=4

cd $SLURM_SUBMIT_DIR

dataset_prefix=$SCRATCH/striped

ibrun  python3 ./scripts/generate_input_spike_trains.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --phase-mod --coords-path=${dataset_prefix}/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --selectivity-path=${dataset_prefix}/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --output-path=${dataset_prefix}/dentate/Full_Scale_Control/DG_input_spike_trains_phasemod_20210521.h5 \
    --value-chunk-size=10000 --chunk-size=10000 --io-size 8 --write-size 20000 \
    --n-trials=3 -v



