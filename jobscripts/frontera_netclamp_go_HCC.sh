#!/bin/bash

#SBATCH -J netclamp_go_HCC # Job name
#SBATCH -o ./results/netclamp_go_HCC.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_go_HCC.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 0:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export DATA_PREFIX=$SCRATCH/striped2/dentate

cd $SLURM_SUBMIT_DIR

ibrun -n 1  python3 network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR.yaml \
    -p HCC -g  1044649 -t 9500 --n-trials 1 --use-coreneuron --dt 0.01 \
    --template-paths $MODEL_HOME/XPPcode:$DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $DATA_PREFIX \
    --config-prefix config \
    --results-path results/netclamp \
    --input-features-path $DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --phase-mod --coords-path "$DATA_PREFIX/Full_Scale_Control/DG_coords_20190717_compressed.h5" \
    --arena-id A --trajectory-id Diag \
    --params-path $SCRATCH/dentate/results/netclamp/network_clamp.optimize.HCC_20210905_145222_NOS02279503.yaml

 
