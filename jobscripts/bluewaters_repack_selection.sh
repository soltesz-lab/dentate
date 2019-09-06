#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=16:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=12:00:00
### set the job name
#PBS -N repack_selection
### set the job stdout and stderr
#PBS -e ./results/repack_selection.$PBS_JOBID.err
#PBS -o ./results/repack_selection.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export prefix=/projects/sciteam/bayj/Full_Scale_Control
export input=$prefix/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_selection.h5
export output=$prefix/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_selection_compressed.h5

aprun h5repack -v -f SHUF -f GZIP=9 -i $input -o $output




