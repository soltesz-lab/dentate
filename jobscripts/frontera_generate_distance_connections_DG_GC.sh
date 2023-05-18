#!/bin/bash
#
#SBATCH -J generate_distance_connections_DG_GC
#SBATCH -o ./results/generate_distance_connections_DG_GC.%j.o
#SBATCH -N 120
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 4:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3/3.9.2
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped2/dentate

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$DATA_PREFIX/Full_Scale_Control/DGC_forest_syns_20230511_compressed.h5 \
    --connectivity-path=$DATA_PREFIX/Full_Scale_Control/DG_GC_connections_20230511.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=40 --cache-size=5 --write-size=250 --value-chunk-size=10000 --chunk-size=10000 -v
