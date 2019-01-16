#!/bin/bash
#PBS -l nodes=512:ppn=16:xe
#PBS -q high
#PBS -l walltime=6:00:00
#PBS -e ./results/generate_GC_distance_connections.$PBS_JOBID.err
#PBS -o ./results/generate_GC_distance_connections.$PBS_JOBID.out
#PBS -N generate_GC_distance_connections
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

ulimit -c unlimited

set -x
cd $PBS_O_WORKDIR


aprun -n 8192 -b -- bwpy-environ -- python2.7 ./scripts/generate_distance_connections.py \
       --config=Full_Scale_Pas.yaml \
       --config-prefix=./config \
       --forest-path=$SCRATCH/Full_Scale_Control/DGC_forest_syns_20181222_compressed.h5 \
       --connectivity-path=$SCRATCH/Full_Scale_Control/DG_GC_connections_20181223.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/Full_Scale_Control/DG_coords_20181223.h5 \
       --coords-namespace=Coordinates \
       --io-size=256 --cache-size=20 --write-size=50 --value-chunk-size=200000 --chunk-size=50000 -v
