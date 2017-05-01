#!/bin/bash
#PBS -l nodes=32:ppn=16:xe
#PBS -q high
#PBS -l walltime=3:00:00
#PBS -e ./results/compute_connectivity.$PBS_JOBID.err
#PBS -o ./results/compute_connectivity.$PBS_JOBID.out
#PBS -N compute_connectivity_reduced_20170501
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
aprun -n 256 -d 2 python ./scripts/compute_DG_connectivity.py
