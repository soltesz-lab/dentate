#!/bin/bash
#PBS -l nodes=16:ppn=16:xe
#PBS -q high
#PBS -l walltime=4:00:00
#PBS -e ./results/interpolate_DG_soma_locations.$PBS_JOBID.err
#PBS -o ./results/interpolate_DG_soma_locations.$PBS_JOBID.out
#PBS -N interpolate_DG_soma_locations_20170505
### set email notification
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export LD_LIBRARY_PATH=/sw/bw/bwpy/0.3.0/python-mpi/usr/lib:/sw/bw/bwpy/0.3.0/python-single/usr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:/projects/sciteam/baef/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH

export PI_HOME=/projects/sciteam/baef
cd $PBS_O_WORKDIR

set -x
aprun -n 256 python ./scripts/interpolate_DG_soma_locations.py \
    --coords-path=/projects/sciteam/baef/Full_Scale_Control/dentate_Sampled_Soma_Locations.h5
