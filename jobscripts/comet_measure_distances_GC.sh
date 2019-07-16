#!/bin/bash
#
#SBATCH -J measure_distances
#SBATCH -o ./results/measure_distances.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH -t 4:00:00
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

export LD_PRELOAD=/opt/openmpi/gnu/ib/lib/libmpi.so
ibrun -np 96 python3.5 ./scripts/measure_distances.py \
    --config=./config/Full_Scale_Basis.yaml --resolution 40 40 10 \
     -i GC \
    --geometry-path=./datasets/dentate_geometry.h5 \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DGC_coords_reindex_20190715.h5 \
    --coords-namespace='Interpolated Coordinates' \
    --io-size=24 \
    -v

