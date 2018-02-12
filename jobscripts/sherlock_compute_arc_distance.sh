#!/bin/bash
#
#SBATCH -J compute_arc_distance
#SBATCH -o ./results/compute_arc_distance.%j.o
#SBATCH -n 96
#SBATCH -t 4:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load mpich/3.1.4/gcc
module load gcc/4.9.1
module load python/2.7.5

export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/model/dentate:$HOME/bin/nrn/lib64/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x

mpirun -np 96 python ./scripts/compute_arc_distance.py \
       --config=./config/Full_Scale_Control.yaml \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_distances_20171109.h5 \
       --coords-namespace=Coordinates \
       -l OML \
       --io-size=24 -v

