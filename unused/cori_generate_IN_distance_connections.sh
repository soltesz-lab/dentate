#!/bin/bash
#
#SBATCH -J generate_IN_distance_connections
#SBATCH -o ./results/generate_IN_distance_connections.%j.o
#SBATCH -N 64
#PBS -N generate_IN_distance_connections
#SBATCH -p regular
#SBATCH -t 3:00:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel
module load python/2.7-anaconda-4.4

set -x

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/cori/2.7-anaconda-4.4/lib/python2.7/site-packages:$PYTHONPATH

srun -n 2048 python2.7 ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20180908.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20180908.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20180717.h5 \
       --coords-namespace=Coordinates \
       --io-size=64 --cache-size=20 --write-size=40 --value-chunk-size=50000 --chunk-size=10000
