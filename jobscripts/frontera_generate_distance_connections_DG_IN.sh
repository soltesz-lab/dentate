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

module load gcc/9.1.0
module load phdf5


export NEURONROOT=$SCRATCH/bin/nrnpython3_gcc9
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/gcc9:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

set -x

export DATA_PREFIX=$SCRATCH/striped2/dentate
 
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_forest_syns_20210920_compressed.h5 \
    --connectivity-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_connections_20211026.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=20 --cache-size=20 --write-size=0 -v

