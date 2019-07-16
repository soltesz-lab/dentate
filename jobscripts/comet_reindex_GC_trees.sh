#!/bin/bash
#
#SBATCH -J reindex_GC_trees
#SBATCH -o ./results/reindex_GC_trees.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=12
#SBATCH -t 5:00:00
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

ibrun -np 384 python3.5 ./scripts/reindex_trees.py \
    --population=GC \
    --types-path=$HOME/model/dentate/datasets/dentate_h5types.h5 \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_extended_compressed_20180224.h5 \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_reindex_20190716.h5 \
    --index-path=$SCRATCH/dentate/Full_Scale_Control/dentate_GC_coords_20190716.h5 \
    --io-size=48 -v
