#!/bin/bash
#
#SBATCH -J generate_gapjunctions
#SBATCH -o ./results/generate_gapjunctions.%j.o
#SBATCH -N 2
#SBATCH -n 56
#SBATCH -p development      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export DATA_PREFIX=$SCRATCH/striped/dentate

set -x


ibrun python3 ./scripts/generate_gapjunctions.py \
    --config=./config/Full_Scale_Basis.yaml \
    --types-path=./datasets/dentate_h5types_gj.h5 \
    --forest-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_forest_syns_20210107_compressed.h5 \
    --connectivity-path=$DATA_PREFIX/Full_Scale_Control/DG_gapjunctions_20210728.h5 \
    --connectivity-namespace="Gap Junctions" \
    --coords-path=$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace="Coordinates" \
    --io-size=24 -v

