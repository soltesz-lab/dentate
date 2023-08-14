#!/bin/bash

#SBATCH -J dentate_cut_slice # Job name
#SBATCH -o ./results/dentate_cut_slice.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate_cut_slice.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 10             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/python3.10:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

results_path=$SCRATCH/dentate/results/cut_slice_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

ibrun python3 ./scripts/cut_slice.py \
    --arena-id=A --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_SynExp3NMDA2_IN_PR.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/striped2/dentate" \
    --output-path=$results_path \
    --io-size=12 \
    --spike-input-path="$SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_spike_trains_phasemod_20220228_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --spike-input-attr="Spike Train" \
    --distance-limits 1900 2000 \
    --write-selection \
    --verbose
