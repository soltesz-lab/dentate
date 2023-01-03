#!/bin/bash
#
#SBATCH -J generate_distance_connections_DG_IN
#SBATCH -o ./results/generate_distance_connections_DG_IN.%j.o
#SBATCH -N 20
#SBATCH -n 960
#SBATCH -p development      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3/3.9.2
module load phdf5/1.10.4


export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

set -x

export DATA_PREFIX=$SCRATCH/striped2/dentate
 
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6
export I_MPI_ADJUST_ALLGATHER=4

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_forest_syns_20211026_compressed.h5 \
    --connectivity-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_connections_20221210.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=20 --cache-size=20 --write-size=0 -v

