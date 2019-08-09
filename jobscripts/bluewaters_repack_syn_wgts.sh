#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=8:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=12:00:00
### set the job name
#PBS -N repack_syns_wgts
### set the job stdout and stderr
#PBS -e ./results/repack_syns_wgts.$PBS_JOBID.err
#PBS -o ./results/repack_syns_wgts.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export input=$SCRATCH/dentate/results/
export output=$SCRATCH/dentate/results/

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output




