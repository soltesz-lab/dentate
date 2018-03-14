#!/bin/bash
#
#SBATCH -J measure_GC_trees
#SBATCH -o ./results/measure_GC_trees.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
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
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 384 python ./scripts/measure_trees.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/model/dgc/Mateos-Aparicio2014 --populations=GC \
              --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_20180306.h5 \
              --io-size=24 --cache-size=$((8 * 1024 * 1024)) \
              --chunk-size=10000 --value-chunk-size=50000 -v
