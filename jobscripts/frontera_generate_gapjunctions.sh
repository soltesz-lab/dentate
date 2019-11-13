#!/bin/bash
#
#SBATCH -J generate_gapjunctions
#SBATCH -o ./results/generate_gapjunctions.%j.o
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -p development      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

#module load python3
#module load phdf5/1.8.16

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

set -x


ibrun python3 ./scripts/generate_gapjunctions.py \
    --config=./config/Full_Scale_Basis.yaml \
    --types-path=./datasets/dentate_h5types_gj.h5 \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_20191112_compressed.h5 \
    --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_gapjunctions_20191112.h5 \
    --connectivity-namespace="Gap Junctions" \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --coords-namespace="Coordinates" \
    --io-size=24 -v

