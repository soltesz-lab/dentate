#!/bin/bash
#
#SBATCH -J compute_arc_distance
#SBATCH -o ./results/compute_arc_distance.%j.o
#SBATCH -N 16
#SBATCH --ntasks-per-node=32
#PBS -N compute_arc_distance
#SBATCH -p regular
#SBATCH -t 4:00:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load PrgEnv-intel 
module unload darshan
module load cray-hdf5-parallel/1.8.16
module load python

set -x

export PYTHONPATH=$HOME/model/dentate:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:$PYTHONPATH


srun -n 512 python ./scripts/compute_arc_distance.py \
    --coords-path=$SCRATCH/dentate/dentate_Full_Scale_Control_coords_20171005.h5 \
    --coords-namespace=Coordinates \
    --io-size=32


