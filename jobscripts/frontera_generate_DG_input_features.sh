#!/bin/bash
#
#SBATCH -J generate_DG_input_features
#SBATCH -o ./results/generate_DG_input_features.%j.o
#SBATCH -p development      # Queue (partition) name
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5/1.10.4

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH

set -x

cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped2/dentate  

export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLREDUCE=2
export I_MPI_ADJUST_BCAST=4
export I_MPI_SHM=off

# -p GC -p MPP -p LPP -p CA3c -p MC -p ConMC
ibrun -n 2000 python3 $HOME/model/dentate/scripts/generate_input_selectivity_features.py \
    --config=Full_Scale_Basis.yaml -p GC --write-size 50000 \
    --config-prefix=./config --use-noise-gen \
    --coords-path=${DATA_PREFIX}/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --output-path=${DATA_PREFIX}/Full_Scale_Control/DG_input_features_20220108.h5 \
    --io-size 2 \
    -v
