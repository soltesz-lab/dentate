#!/bin/bash
#PBS -l nodes=8:ppn=16:xe
#PBS -q high
#PBS -l walltime=4:00:00
#PBS -e ./results/compute_DG_PP_feature_selectivity.$PBS_JOBID.err
#PBS -o ./results/compute_DG_PP_feature_selectivity.$PBS_JOBID.out
#PBS -N compute_DG_PP_feature_selectivity
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

aprun -n 128 python ./scripts/compute_DG_PP_feature_selectivity.py \
              --coords-path=/projects/sciteam/baef/Full_Scale_Control/dentate_Full_Scale_Control_coords_20171006.h5 \
              --io-size=16
