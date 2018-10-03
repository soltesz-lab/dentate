#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH -N 32
#SBATCH --ntasks-per-node=32
#PBS -N distribute_GC_synapses
#SBATCH -p regular
#SBATCH -t 4:00:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#


module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel
module load python/2.7-anaconda-4.4

set -x

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/cori/2.7-anaconda-4.4/lib/python2.7/site-packages:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.6


srun -n 1024 python ./scripts/distribute_synapse_locs.py \
 --distribution=poisson \
 --config=./config/Full_Scale_Control.yaml  \
 --template-path=$HOME/model/dgc/Mateos-Aparicio2014 \
 --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_20180425.h5 \
 --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20180812.h5 \
 --populations=GC \
 --io-size=64 --cache-size=50 \
 --chunk-size=10000 --value-chunk-size=50000
