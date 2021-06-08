#!/bin/bash

#SBATCH -J netclamp_go_GC_features # Job name
#SBATCH -o ./results/netclamp_go_GC_features.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_go_GC_features.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 2             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate


cd $SLURM_SUBMIT_DIR

export dataset_prefix=$SCRATCH/striped/dentate

python3  network_clamp.py go  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
         -p GC -g 317622  -t 9500 \
         --template-paths templates:$HOME/src/model/DGC/Mateos-Aparicio2014 \
         --dataset-prefix $dataset_prefix \
         --input-features-path $dataset_prefix/DG_input_features_20200910_compressed.h5 \
         --input-features-namespaces 'Place Selectivity' \
         --input-features-namespaces 'Grid Selectivity' \
         --input-features-namespaces 'Constant Selectivity' \
         --phase-mod \
         --coords-path $dataset_prefix/DG_coords_20190717_compressed.h5 \
         --arena-id A --trajectory-id Diag  --n-trials 1 \
         --recording-profile 'Network clamp default' \
         --config-prefix config  \
         --results-path results/netclamp

