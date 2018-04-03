#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q normal
#PBS -l walltime=1:30:00
#PBS -e ./results/generate_DG_GC_log_normal_weights_as_cell_attr.$PBS_JOBID.err
#PBS -o ./results/generate_DG_GC_log_normal_weights_as_cell_attr.$PBS_JOBID.out
#PBS -N generate_DG_GC_log_normal_weights
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export ATP_ENABLED=1 
export PYTHONPATH=$HOME/model:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

set -x
cd $PBS_O_WORKDIR



aprun -n 1024 -b -- bwpy-environ -- python2.7 $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s LPP -s MPP -s MC \
    --config=./config/Full_Scale_Control.yaml \
    --weights-path=$SCRATCH/Full_Scale_Control/DG_GC_forest_syns_weights_20180329.h5 \
    --connections-path=$SCRATCH/Full_Scale_Control/DG_GC_connections_compressed_20180319.h5 \
    --io-size=256 --cache-size=1 --value-chunk-size=100000 --chunk-size=20000 -v
