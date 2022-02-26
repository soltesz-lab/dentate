#!/bin/bash
#
#SBATCH -J measure_trees_DG_GC
#SBATCH -o ./results/measure_trees_DG_GC.%j.o
#SBATCH -N 10
#SBATCH --ntasks-per-node=56
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

set -x

ibrun python3 ./scripts/measure_trees.py \
    --config=./config/Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates \
    -i GC \
    --forest-path=$SCRATCH/striped2/dentate/Full_Scale_Control/DGC_forest_normalized_20200628_compressed.h5 \
    --output-path=$SCRATCH/dentate/results/measure_trees_DG_GC.h5 \
    --io-size=8 -v

