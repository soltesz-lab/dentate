#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=8:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N nrnpy_test
### set the job stdout and stderr
#PBS -e ./results/nrnpy.$PBS_JOBID.err
#PBS -o ./results/nrnpy.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A bayj


module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

set -x

export SCRATCH=/projects/sciteam/bayj
export NEURONROOT=$SCRATCH/nrn
export PYTHONPATH=$NEURONROOT/lib/python:$PYTHONPATH

echo PYTHONPATH is $PYTHONPATH

#cd $PBS_O_WORKDIR

aprun -n 8 -b -- bwpy-environ -- python2.7 ./tests/testmpi.py
