#!/bin/bash
#
#SBATCH -J measure_GC_distances
#SBATCH -o ./results/measure_GC_distances.%j.o
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH -q regular
#SBATCH -t 4:00:00
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
export LD_PRELOAD=/lib64/libreadline.so.6


srun -n 320 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates \
              -i GC \
              --coords-path=$SCRATCH/dentate/Full_Scale_Control//DG_coords_20180605.h5 \
              --io-size=8 -v
