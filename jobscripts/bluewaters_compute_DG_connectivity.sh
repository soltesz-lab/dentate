#!/bin/bash
#PBS -l nodes=128:ppn=16:xe
#PBS -q high
#PBS -l walltime=5:00:00
#PBS -e ./results/compute_connectivity.$PBS_JOBID.err
#PBS -o ./results/compute_connectivity.$PBS_JOBID.out
#PBS -N compute_connectivity_20170508
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
#source $HOME/.bash_profile
cd $PBS_O_WORKDIR

set -x
aprun -n 1024 -d 2 python ./scripts/compute_DG_connectivity.py \
  --forest-path=/projects/sciteam/baef/Full_Scale_Control/DGC_forest_connectivity_20170508.h5 \
  --coords-path=/projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_Control_coords_20170508.h5 \
  --io-size=256 --chunk-size=10000 --value-chunk-size=50000

