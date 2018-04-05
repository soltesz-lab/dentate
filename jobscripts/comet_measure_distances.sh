#!/bin/bash
#
#SBATCH -J measure_distances
#SBATCH -o ./results/measure_distances.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=12
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

ibrun -np 192 python ./scripts/measure_distances.py \
    --config=./config/Full_Scale_Control.yaml \
     -i GC -i MPP -i LPP -i AAC -i BC -i MC -i HC -i HCC -i NGFC -i MOPP -i IS \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_cells_20180305.h5 \
    --coords-namespace=Coordinates \
    --io-size=24 \
    -v
