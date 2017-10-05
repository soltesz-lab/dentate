#!/bin/bash
#
#SBATCH -J generate_distance_connections
#SBATCH -o ./results/generate_distance_connections.%j.o
#SBATCH -n 128
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python/2.7.5
module load mpich/3.1.4/gcc

export PATH=$PI_HOME/neuron/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PI_HOME/neuron/nrn/lib64/python:$PYTHONPATH
export LD_LIBRARY_PATH=$PI_HOME/hdf5/lib:$LD_LIBRARY_PATH

set -x

mpirun -np 128 python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/dentate/DG_test_forest_syns_20170927.h5 \
       --connectivity-path=$SCRATCH/dentate/DG_test_connections_20171001.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/dentate_test_coords_20170929.h5 \
       --coords-namespace=Coordinates

