#!/bin/bash
#
#SBATCH -J reindex_GC_trees
#SBATCH -o ./results/reindex_GC_trees.%j.o
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 3:00:00
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

ibrun -np 12 python3 ./scripts/reindex_trees.py \
    --population=GC \
    --sample-count=1000 \
    --types-path=./datasets/dentate_h5types.h5 \
    --forest-path=$SCRATCH/dentate/Test_GC_1000/DGC_forest_20190622.h5 \
    --output-path=$SCRATCH/dentate/Test_GC_1000/DGC_forest_reindex_20190622.h5 \
    --index-path=$SCRATCH/dentate/Test_GC_1000/DGC_coords_20190622.h5 \
    --io-size=4 -v
