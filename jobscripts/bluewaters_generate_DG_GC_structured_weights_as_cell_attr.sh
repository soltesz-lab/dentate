#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q high
#PBS -l walltime=2:00:00
#PBS -e ./results/generate_DG_GC_structured_weights.$PBS_JOBID.err
#PBS -o ./results/generate_DG_GC_structured_weights.$PBS_JOBID.out
#PBS -N generate_DG_GC_structured_weights
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


aprun -n 1024 -b -- bwpy-environ -- python2.7 $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
 -d GC -s LPP -s MPP \
 --config=./config/Full_Scale_Control.yaml \
 --stimulus-path=$SCRATCH/Full_Scale_Control/DG_PP_features_20180326.h5 \
 --stimulus-namespace='Vector Stimulus' \
 --initial-weights-namespace='Weights' --structured-weights-namespace='Structured Weights' \
 --weights-path=$SCRATCH/Full_Scale_Control/DGC_forest_syns_weights_compressed_20180401.h5 \
 --connections-path=$SCRATCH/Full_Scale_Control/DG_GC_connections_compressed_20180319.h5 \
 --trajectory-id=0 \
 --io-size=256 --value-chunk-size=100000 --chunk-size=20000 --write-size=25 -v 
