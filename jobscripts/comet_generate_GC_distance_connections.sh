#!/bin/bash
#
#SBATCH -J generate_GC_distance_connections
#SBATCH -o ./results/generate_GC_distance_connections.%j.o
#SBATCH --nodes=36
#SBATCH --ntasks-per-node=20
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
export PYTHONPATH=$HOME/model/dentate:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 720 python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20171031_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20171120.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_distances_20171109.h5 \
       --coords-namespace=Coordinates \
       --distances-namespace="Arc Distance Layer OML" \
       --io-size=256 --cache-size=1 --value-chunk-size=50000 --chunk-size=10000 --quick

