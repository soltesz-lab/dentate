#!/bin/bash
#
#SBATCH -J interpolate_GC_soma_locations
#SBATCH -o ./results/interpolate_GC_soma_locations.%j.o
#SBATCH --nodes=4
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

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so

set -x

ibrun -np 48 python ./scripts/interpolate_forest_soma_locations.py \
    --config=./config/Test_GC_1000.yaml \
    --forest-path=$SCRATCH/dentate/Test_GC_1000/DG_GC_forest_20181222.h5 \
    --coords-path=$SCRATCH/dentate/Test_GC_1000/dentate_GC_coords_20180520.h5 \
    -i GC --reltol=5 \
    --io-size=4 -v
