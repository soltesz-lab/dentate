#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q normal
#PBS -l walltime=2:30:00
#PBS -e ./results/generate_IN_distance_connections.$PBS_JOBID.err
#PBS -o ./results/generate_IN_distance_connections.$PBS_JOBID.out
#PBS -N generate_IN_distance_connections
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

ulimit -c unlimited

set -x
cd $PBS_O_WORKDIR


aprun -n 1024 -b -- bwpy-environ -- python ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/Full_Scale_Control/DG_IN_forest_syns_20180304.h5 \
       --connectivity-path=$SCRATCH/Full_Scale_Control/DG_IN_connections_20180323.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/Full_Scale_Control/DG_cells_20180305.h5 \
       --coords-namespace=Coordinates \
       --resample-volume=2 \
       --io-size=128 --cache-size=1 --write-size=25 -v
