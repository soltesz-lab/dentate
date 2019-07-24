#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=16
#SBATCH -t 8:00:00
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

ibrun -np 1024 python3 ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates --populations=GC \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_reindex_20190717_compressed.h5 \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20190717.h5 \
    --io-size=256 --write-size=10 --cache-size=$((2 * 1024 * 1024)) \
    --chunk-size=50000 --value-chunk-size=100000 -v
