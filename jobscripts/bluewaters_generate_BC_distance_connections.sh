#!/bin/bash
#PBS -l nodes=12:ppn=16:xe
#PBS -q high
#PBS -l walltime=3:00:00
#PBS -e ./results/generate_BC_distance_connections.$PBS_JOBID.err
#PBS -o ./results/generate_BC_distance_connections.$PBS_JOBID.out
#PBS -N generate_BC_distance_connections
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi


export ATP_ENABLED=1 
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

ulimit -c unlimited

set -x
cd $PBS_O_WORKDIR


aprun -n 192 -b -- bwpy-environ -- python2.7 ./scripts/generate_distance_connections.py \
       --config=./config/Full_Scale_Control.yaml \
       --forest-path=$SCRATCH/Full_Scale_Control/BC_forest_syns_20180630.h5 \
       --connectivity-path=$SCRATCH/Full_Scale_Control/DG_BC_connections_20180717.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/Full_Scale_Control/DG_coords_20180717.h5 \
       --coords-namespace=Coordinates \
       --io-size=16 --cache-size=1 --write-size=10 -v
