#!/bin/bash

#SBATCH -J dentate_Slice_neg2000_neg1900um_GC_Aradi_Sat_SLN_IN_PR_Diag # Job name
#SBATCH -o ./results/dentate.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 10             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load python3
module load phdf5/1.10.4

set -x

export NEURONROOT=$HOME/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export DATA_PREFIX=$SCRATCH/striped2/dentate

export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

results_path=$SCRATCH/dentate/results/Slice_neg2000_neg1900um_GC_Aradi_SLN_IN_PR_Diag_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin

ibrun -n 168 env PYTHONPATH=$PYTHONPATH python3 ./scripts/main.py  \
    --config-file=Network_Clamp_Slice_neg2000_neg1900um_IN_PR.yaml  \
    --arena-id=A --trajectory-id=Diag \
    --template-paths=templates \
    --dataset-prefix="$SCRATCH/striped2/dentate" \
    --results-path=$results_path \
    --spike-input-path="$DATA_PREFIX/Slice/dentatenet_Slice_SLN_neg2000_neg1900um_20220108_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --spike-input-attr='Spike Train' \
    --microcircuit-inputs \
    --io-size=1 \
    --use-cell-attr-gen \
    --tstop=1000 \
    --v-init=-75 \
    --results-write-time=600 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=0.49 \
    --dt 0.025 --use-coreneuron \
    --verbose

