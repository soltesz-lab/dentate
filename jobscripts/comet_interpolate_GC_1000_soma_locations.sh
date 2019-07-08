#!/bin/bash
#
#SBATCH -J interpolate_GC_soma_locations
#SBATCH -o ./results/interpolate_GC_soma_locations.%j.o
#SBATCH -p shared
#SBATCH --ntasks=12
#SBATCH -t 0:30:00
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
ulimit -c unlimited

set -x

ibrun -np 1 python3 ./scripts/interpolate_forest_soma_locations.py \
    --config-prefix=./config \
    --config=Test_GC_1000.yaml \
    --resolution 40 40 10 \
    --forest-path=$SCRATCH/dentate/Test_GC_1000/DGC_forest_20190622.h5 \
    --coords-path=$SCRATCH/dentate/Test_GC_1000/DGC_coords_20190622.h5 \
    -i GC --reltol=5 \
    --io-size=1 -v
