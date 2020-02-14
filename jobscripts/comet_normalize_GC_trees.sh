#!/bin/bash
#
#SBATCH -J normalize_GC_trees
#SBATCH -o ./results/normalize_GC_trees.%j.o
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=24
#SBATCH -t 5:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#
. $HOME/comet_env.sh

ulimit -c unlimited

set -x

ibrun -v python3 ./scripts/normalize_trees.py \
    --population=GC \
    --config-prefix=$HOME/model/dentate/config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_reindex_20190717_compressed.h5 \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_normalized_20200214.h5 \
    --io-size=48 -v


