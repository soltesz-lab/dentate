#!/bin/bash
#
#SBATCH -J generate_distance_connections_IN
#SBATCH -o ./results/generate_distance_connections_IN.%j.o
#SBATCH -N 20
#SBATCH -n 960
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load phdf5


export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

set -x


export I_MPI_EXTRA_FILESYSTEM=enable
export I_MPI_EXTRA_FILESYSTEM_LIST=lustre

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_IN_forest_syns_20210107.h5 \
    --connectivity-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_IN_connections_20210107.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=20 --cache-size=20 --write-size=0 -v

