#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
#SBATCH -p normal      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load phdf5

export NEURONROOT=$HOME/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH


ibrun python3  ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates --populations=GC \
    --forest-path=$SCRATCH/striped/dentate/Full_Scale_Control/DGC_forest_normalized_20200628_compressed.h5 \
    --output-path=$SCRATCH/striped/dentate/Full_Scale_Control/DGC_forest_syns_20201217.h5 \
    --io-size=40 --write-size=10 --cache-size=$((2 * 1024 * 1024)) \
    --chunk-size=50000 --value-chunk-size=100000 -v
