#!/bin/bash
#
#SBATCH -J generate_distance_connections
#SBATCH -o ./results/generate_distance_connections.%j.o
#SBATCH -n 48
#SBATCH -t 8:00:00
#SBATCH --mem-per-cpu=10G
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

mpirun -np 48 python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/DG_test_forest_syns_20170927.h5 \
       --connectivity-path=$SCRATCH/dentate/DG_test_connections_20171012.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/dentate_Full_Scale_Control_coords_20171005.h5 \
       --coords-namespace=Coordinates \
       --io-size=4

