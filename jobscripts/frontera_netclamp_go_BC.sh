#!/bin/bash

#SBATCH -J netclamp_go_BC # Job name
#SBATCH -o ./results/netclamp_go_BC.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_go_BC.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 0:10:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5/1.10.4

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export UCX_TLS="knem,dc_x"
export DATA_PREFIX=$SCRATCH/striped2/dentate


gid=1042795    
gid=1040200
gid=1041000

mpirun -n 1 python3 network_clamp.py go \
    --config-file Network_Clamp_GC_Exc_Sat_SLN_IN_PR.yaml \
    --template-paths=templates \
    -p BC -g $gid -t 9500 --dt 0.01 --use-coreneuron \
    --dataset-prefix $DATA_PREFIX \
    --config-prefix config \
    --input-features-path $DATA_PREFIX/Full_Scale_Control/DG_input_features_20220131_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --arena-id A --trajectory-id Diag \
    --phase-mod --coords-path "$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5" \
    --results-path results/netclamp


