#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=16:ppn=16:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=3:00:00
### set the job name
#PBS -N vertex_metrics
### set the job stdout and stderr
#PBS -e ./results/vertex_metrics.$PBS_JOBID.err
#PBS -o ./results/vertex_metrics.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export prefix=/projects/sciteam/baqc/Full_Scale_Control
export input=$prefix/DG_GC_connections_20180813_compressed.h5

aprun  -n 256 $HOME/src/neuroh5/build/neurograph_vertex_metrics --indegree --outdegree -i 128 \
    $input



