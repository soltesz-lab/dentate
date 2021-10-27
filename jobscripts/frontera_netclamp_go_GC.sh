#!/bin/bash

#SBATCH -J netclamp_go_GC # Job name
#SBATCH -o ./results/netclamp_go_GC.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_go_GC.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load gcc/9.1.0
module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_gcc9
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/gcc9:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export UCX_TLS="knem,dc_x"
export DATA_PREFIX=$SCRATCH/striped2/dentate

# Population offsets:
#
# GC:    0
# MC:    1000000
# HC:    1030000
# BC:    1039000
# AAC:   1042800
# HCC:   1043250
# NGFC:  1044650
# IS:    1049650
# MOPP:  1052650
# MPP:   1056650
# LPP:   1094650
# CA3c:  1128650,
# ConMC: 1195650

gid=197226    

mpirun -n 1 python3 network_clamp.py go \
    --config-file Full_Scale_GC_Aradi_SLN_IN_PR.yaml \
    --template-paths=templates \
    -p GC -g $gid -t 9500 --dt 0.01 --use-coreneuron \
    --dataset-prefix $DATA_PREFIX \
    --config-prefix config \
    --input-features-path $DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --results-path results/netclamp
