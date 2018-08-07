#!/bin/bash
#PBS -l nodes=128:ppn=16:xe
#PBS -q high
#PBS -l walltime=4:00:00
#PBS -e ./results/distribute_GC_synapses.$PBS_JOBID.err
#PBS -o ./results/distribute_GC_synapses.$PBS_JOBID.out
#PBS -N distribute_GC_synapses
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

set -x
cd $PBS_O_WORKDIR

aprun -n 2048 -b -- bwpy-environ -- python2.7 ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config=./config/Full_Scale_Control.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014 --populations=GC \
    --forest-path=$SCRATCH/Full_Scale_Control/DGC_forest_20180425.h5 \
    --output-path=$SCRATCH/Full_Scale_Control/DGC_forest_syns_20180807.h5 \
    --io-size=256 --cache-size=$((8 * 1024 * 1024)) \
    --chunk-size=50000 --value-chunk-size=200000 -v

