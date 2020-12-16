#!/bin/bash

#SBATCH -J dentate_cut_slice # Job name
#SBATCH -o ./results/dentate_cut_slice.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate_cut_slice.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 30             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load intel/18.0.5
module load python3
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3/intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

results_path=$SCRATCH/dentate/results/cut_slice_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path


export PYTHON=`which python3`

ibrun python3 ./scripts/cut_slice.py \
    --arena-id=A --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/striped/dentate" \
    --output-path=$results_path \
    --io-size=96 \
    --spike-input-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h\
5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --distance-limits -2000 -1500 \
    --write-selection \
    --verbose
