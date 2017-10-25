#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=8:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=12:00:00
### set the job name
#PBS -N dentate_repack
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export prefix=/projects/sciteam/baef/Full_Scale_Control
export input=$prefix/DGC_forest_syns_20171024.h5
export output=$prefix/DGC_forest_syns_20171024_compressed.h5

aprun h5repack -v -i $input -o $output




