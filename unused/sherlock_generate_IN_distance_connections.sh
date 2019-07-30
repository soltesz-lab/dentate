#!/bin/bash
#
#SBATCH -J generate_IN_distance_connections
#SBATCH -o ./results/generate_IN_distance_connections.%j.o
#SBATCH -n 384
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python/2.7.5
module load mpich/3.1.4/gcc
module load gcc/4.9.1

export PYTHONPATH=$HOME/model/dentate:$HOME/bin/nrn/lib64/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x

mpirun -np 384 python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20171101.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20171101.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_20171005.h5 \
       --coords-namespace=Coordinates \
       --io-size=64 --cache-size=1 --quick

