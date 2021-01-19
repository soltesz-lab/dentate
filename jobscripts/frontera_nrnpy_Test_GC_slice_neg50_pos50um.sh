#!/bin/bash
#
#SBATCH -J dentate_Test_GC_slice_neg50_pos50um
#SBATCH -o ./results/dentate_Test_GC_slice_neg50_pos50um.%j.o
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=56
#SBATCH -p normal
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load intel/18.0.5 
module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

#export I_MPI_EXTRA_FILESYSTEM=enable
export I_MPI_ADJUST_ALLGATHER=4
export I_MPI_ADJUST_ALLGATHERV=4
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2

results_path=$SCRATCH/striped/dentate/results/Test_GC_slice_neg50_pos50um_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun python3 ./scripts/main.py  \
    --arena-id=A --trajectory-id=Diag \
    --config-file=Network_Clamp_Slice_neg50_pos50um_IN_Izh_opt20201221.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH/striped/dentate" \
    --results-path=$results_path \
    --io-size=8 \
    --tstop=9500 \
    --v-init=-75 \
    --max-walltime-hours=1.9 \
    --spike-input-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --spike-input-attr='Spike Train' \
    --microcircuit-inputs \
    --checkpoint-interval 0. \
    --recording-fraction 0.04 \
    --use-coreneuron \
    --verbose

