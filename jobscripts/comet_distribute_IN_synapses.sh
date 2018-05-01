#!/bin/bash
#
#SBATCH -J distribute_IN_synapses
#SBATCH -o ./results/distribute_IN_synapses.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 24 python ./scripts/distribute_synapse_locs.py \
              --distribution=poisson \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$PWD/templates \
              -i IS \
              --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20180304.h5 \
              --output-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20180304.h5 \
              --io-size=2 -v
