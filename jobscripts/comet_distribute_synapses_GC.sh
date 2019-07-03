#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 1536 python ./scripts/distribute_synapse_locs.py \
              --distribution=poisson \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/model/dgc/Mateos-Aparicio2014 --populations=GC \
              --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_20180425.h5 \
              --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20180425.h5 \
              --io-size=128 --cache-size=$((8 * 1024 * 1024)) \
              --chunk-size=50000 --value-chunk-size=100000 -v
