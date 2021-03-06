#!/bin/bash

#SBATCH -J dentate_sample_extent_proximal_pf # Job name
#SBATCH -o ./results/dentate_sample_extent_proximal_pf.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate_sample_extent_proximal_pf.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 1:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

results_path=$SCRATCH/dentate/results/sample_extent_proximal_pf_$SLURM_JOB_ID
export results_path
mkdir -p $results_path

cd $SLURM_SUBMIT_DIR

ibrun -np 16  python3 ./scripts/sample_extent.py \
    --arena-id='A' --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/striped/dentate" \
    --output-path=$results_path \
    --spike-input-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h\
5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --input-features-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --output-path=${results_path} \
    --bin-sample-proximal-pf \
    --bin-sample-count 10 \
    --distance-bin-extent=500. \
    --write-selection \
    -i GC -i MC \
    --io-size 4 \
    --verbose
