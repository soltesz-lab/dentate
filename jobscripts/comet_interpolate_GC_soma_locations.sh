#!/bin/bash
#
#SBATCH -J interpolate_GC_soma_locations
#SBATCH -o ./results/interpolate_GC_soma_locations.%j.o
#SBATCH --nodes=32
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

ibrun -np 768 python ./scripts/interpolate_forest_soma_locations.py \
    --config=./config/Full_Scale_Control.yaml \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_extended_compressed_20180215.h5 \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_GC_coords_20180222.h5 \
    -i GC --rotate=-35 \
    --io-size=24 -v
