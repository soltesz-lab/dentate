#!/bin/bash
#PBS -l nodes=64:ppn=16:xe
#PBS -q high
#PBS -l walltime=12:00:00
#PBS -e ./results/compute_GC_syn_locs.$PBS_JOBID.err
#PBS -o ./results/compute_GC_syn_locs.$PBS_JOBID.out
#PBS -N compute_GC_syn_locs
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

source $HOME/.bash_profile
cd $PBS_O_WORKDIR
#export PI_HOME=/projects/sciteam/baef

set -x
aprun -n 1024 -d 2 python ./scripts/compute_GC_synapse_locs.py
