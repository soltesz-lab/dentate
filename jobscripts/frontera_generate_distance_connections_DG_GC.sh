#!/bin/bash
#
#SBATCH -J generate_distance_connections_GC
#SBATCH -o ./results/generate_distance_connections_GC.%j.o
#SBATCH -N 40
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH

cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control  
 

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$SCRATCH/striped/dentate/Full_Scale_Control/DGC_forest_syns_20201217_compressed.h5 \
    --connectivity-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_GC_connections_20201217.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=40 --cache-size=1 --write-size=0 -v

