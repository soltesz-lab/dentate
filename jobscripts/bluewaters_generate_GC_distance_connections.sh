#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q normal
#PBS -l walltime=6:00:00
#PBS -e ./results/generate_GC_distance_connections.$PBS_JOBID.err
#PBS -o ./results/generate_GC_distance_connections.$PBS_JOBID.out
#PBS -N generate_distance_connections
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi


export ATP_ENABLED=1 
export PYTHONPATH=$HOME/model:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

ulimit -c unlimited

set -x
cd $PBS_O_WORKDIR


aprun -n 1024 -b -- bwpy-environ -- python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/Full_Scale_Control/DGC_forest_syns_compressed_20180306.h5 \
       --connectivity-path=$SCRATCH/Full_Scale_Control/DG_GC_connections_20180323.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/Full_Scale_Control/DG_cells_20180305.h5 \
       --coords-namespace=Coordinates \
       --resample-volume=2 \
       --io-size=256 --cache-size=1 --write-size=25 --value-chunk-size=200000 --chunk-size=20000 -v
