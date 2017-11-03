#!/bin/bash
#PBS -l nodes=128:ppn=16:xe
#PBS -q high
#PBS -l walltime=4:00:00
#PBS -e ./results/distribute_GC_synapses.$PBS_JOBID.err
#PBS -o ./results/distribute_GC_synapses.$PBS_JOBID.out
#PBS -N distribute_GC_synapses
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export LD_LIBRARY_PATH=/sw/bw/bwpy/0.3.0/python-mpi/usr/lib:/sw/bw/bwpy/0.3.0/python-single/usr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/model/dentate:$HOME/bin/nrn/lib/python:/projects/sciteam/baef/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH

set -x
cd $PBS_O_WORKDIR

aprun -n 2048 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/model/dgc/Mateos-Aparicio2014 --populations=GC \
              --forest-path=/projects/sciteam/baef/Full_Scale_Control/DGC_forest_syns_20171031.h5 \
              --io-size=256 --cache-size=$((8 * 1024 * 1024)) \
              --chunk-size=10000 --value-chunk-size=50000
