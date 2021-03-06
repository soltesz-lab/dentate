#!/bin/bash
#
#SBATCH -J generate_distance_connections_DG_GC
#SBATCH -o ./results/generate_distance_connections_DG_GC.%j.o
#SBATCH -N 40
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 4:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH

cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control  
 
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$SCRATCH/striped/dentate/Full_Scale_Control/DGC_forest_syns_20210106_compressed.h5 \
    --connectivity-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_GC_connections_20210107.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=40 --cache-size=5 --write-size=250 --value-chunk-size=10000 --chunk-size=10000 -v


