#!/bin/bash
#SBATCH -J generate_input_spike_trains_DG
#SBATCH -o ./results/generate_input_spike_trains_DG.%j.o
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 16             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load gcc/9.1.0
module load python3
module load phdf5/1.10.4

export NEURONROOT=$SCRATCH/bin/nrnpython3_gcc9
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/gcc9:$PYTHONPATH

set -x

export DATASET_PREFIX=$SCRATCH/striped2  

export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=4
export I_MPI_PIN_DOMAIN=auto
export I_MPI_PLATFORM=auto


cd $SLURM_SUBMIT_DIR

ibrun  python3 ./scripts/generate_input_spike_trains.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --phase-mod --coords-path=${DATASET_PREFIX}/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --selectivity-path=${DATASET_PREFIX}/dentate/Full_Scale_Control/DG_input_features_20220105.h5 \
    --output-path=${DATASET_PREFIX}/dentate/Full_Scale_Control/DG_input_spike_trains_phasemod_20220106.h5 \
    --value-chunk-size=10000 --chunk-size=10000 --io-size 4 --write-size 50000 \
    --n-trials=3 -v



