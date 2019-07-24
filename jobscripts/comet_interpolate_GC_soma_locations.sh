#!/bin/bash
#
#SBATCH -J interpolate_GC_soma_locations
#SBATCH -o ./results/interpolate_GC_soma_locations.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=12
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x

ibrun -np 384 python3.5 ./scripts/interpolate_forest_soma_locations.py \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --resolution 40 40 10 \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_extended_compressed_20180224.h5 \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_GC_coords_20190717.h5 \
    -i GC --reltol=5 \
    --io-size=24 -v
