#!/bin/bash

#SBATCH -J dentate_sample_cells # Job name
#SBATCH -o ./results/dentate_sample_cells.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate_sample_cells.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 20             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 1:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load python3
module load phdf5/1.10.4

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export HDF5_EXT_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control

results_path=$SCRATCH/dentate/results/sample_cells_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

ibrun -n 8 python3 ./scripts/sample_cells.py \
    --arena-id=A --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/striped/dentate" \
    --output-path=$results_path \
    --spike-input-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h\
5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --input-features-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --selection-path=pf_rate_selection_20210114.dat \
    --verbose
