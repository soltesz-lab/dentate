#!/bin/bash
#
#SBATCH -J generate_GC_distance_connections
#SBATCH -o ./results/generate_GC_distance_connections.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/model/dentate:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 1536 python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20171024_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20171029.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/dentate_Full_Scale_Control_coords_20171005.h5 \
       --coords-namespace=Coordinates \
       --io-size=128 --cache-size=1 --value-chunk-size=50000 --chunk-size=10000 --quick

