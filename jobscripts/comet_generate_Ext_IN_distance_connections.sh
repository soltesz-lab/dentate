#!/bin/bash
#
#SBATCH -J generate_Ext_IN_distance_connections
#SBATCH -o ./results/generate_Ext_IN_distance_connections.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x


ibrun -np 240 python ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Full_Scale_Ext.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20190124_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_Ext_IN_connections_20190124.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190122.h5 \
       --coords-namespace=Coordinates \
       --io-size=12 --cache-size=20 --write-size=40 -v

