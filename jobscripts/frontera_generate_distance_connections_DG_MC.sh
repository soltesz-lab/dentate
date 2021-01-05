#!/bin/bash
#
#SBATCH -J generate_distance_connections_MC
#SBATCH -o ./results/generate_distance_connections_MC.%j.o
#SBATCH -N 10
#SBATCH -n 560
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 1:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load python3
module load phdf5

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH

set -x


cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control  

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_MC_forest_syns_20200708_compressed.h5 \
    --connectivity-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_MC_connections_20200708.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace=Coordinates \
    --io-size=12 --cache-size=20 --write-size=40 -v

