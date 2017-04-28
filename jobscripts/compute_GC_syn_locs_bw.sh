#!/bin/bash
#PBS -l nodes=512:ppn=16:xe
#PBS -q high
#PBS -l walltime=4:00:00
#PBS -e ./results/compute_GC_syn_locs.$PBS_JOBID.err
#PBS -o ./results/compute_GC_syn_locs.$PBS_JOBID.out
#PBS -N compute_GC_syn_locs
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

#export PI_HOME=/projects/sciteam/baef

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

set -x
cd $PBS_O_WORKDIR

aprun -n 8192 -d 2 python ./scripts/compute_GC_synapse_locs.py
