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
module unload intel
module load gnu
module load mvapich2_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 24 python3 ./scripts/interpolate_forest_soma_locations.py \
    --config-prefix=./config \
    --config=Test_GC_1000.yaml \
    --forest-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_forest_20190612.h5 \
    --coords-path=$SCRATCH/dentate/Test_GC_1000/DG_GC_1000_coords_20180618.h5 \
    -i GC --reltol=5 \
    --io-size=8 -v
