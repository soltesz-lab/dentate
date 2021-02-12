#!/bin/bash
#SBATCH -J generate_input_spike_trains_DG
#SBATCH -o ./results/generate_input_spike_trains_DG.%j.o
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 4             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load intel/18.0.5
module load python3
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so

cd $SLURM_SUBMIT_DIR

dataset_prefix=$SCRATCH/striped

ibrun  python3 ./scripts/generate_input_spike_trains.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --selectivity-path=${dataset_prefix}/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --output-path=${dataset_prefix}/dentate/Full_Scale_Control/DG_input_spike_trains_20200910.h5 \
    --n-trials=3 -v



