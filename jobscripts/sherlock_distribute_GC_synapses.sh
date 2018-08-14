#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH -n 128
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load openmpi/3.0.0 hdf5 readline python/2.7.13 

export PATH=$HOME/.local/bin:$HOME/bin/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

set -x

mpirun -np 128 python ./scripts/distribute_synapse_locs.py \
 --distribution=poisson \
 --config=./config/Full_Scale_Control.yaml  \
 --template-path=$HOME/model/dgc/Mateos-Aparicio2014 \
 --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_20180425.h5 \
 --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20180807.h5 \
 --populations=GC \
 --io-size=16 --cache-size=10 \
 --chunk-size=50000 --value-chunk-size=100000
