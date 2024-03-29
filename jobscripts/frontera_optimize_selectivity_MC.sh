#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 12             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5/1.10.4

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
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
export I_MPI_ADJUST_ALLREDUCE=4

export FI_MLX_ENABLE_SPAWN=1
export I_MPI_HYDRA_TOPOLIB=ipl
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off
export I_MPI_HYDRA_BRANCH_COUNT=0
export FI_PROVIDER=verbs
#export UCX_TLS="dc_x,knem"

#cd $SLURM_SUBMIT_DIR

mkdir -p $SCRATCH/dentate/results/netclamp/MC_20220202

export nworkers=$((12 * 24))

if test "$3" == ""; then
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR.yaml \
    -p MC -t 9500 -g $1  --n-trials 1 --trial-regime mean --problem-regime every --nprocs-per-worker=1 \
    --n-epochs=10 --n-initial=200 --num-generations=400 --population-size=800 --resample-fraction 0.5 \
    --template-paths $DG_HOME/templates \
    --dataset-prefix $SCRATCH/striped2/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/MC_20220202 \
    --config-prefix config  --param-config-name "$2" --selectivity-config-name CA3c \
    --arena-id A --trajectory-id Diag --use-coreneuron \
    --target-features-path $SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_features_20220131_compressed.h5 \
    --target-features-namespace 'Place Selectivity' \
    --input-features-path $SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_features_20220131_compressed.h5 \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' 
else
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_PR.yaml \
    -p MC -t 9500 -g $1 --n-trials 1 --trial-regime mean --problem-regime every --nprocs-per-worker=1 \
     --n-iter=2 --n-initial=200 --num-generations=400 --population-size=800 --resample-fraction 0.84 \
    --template-paths $DG_HOME/templates \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/MC_20220202 \
    --results-file "$3" \
    --config-prefix config --param-config-name "$2" --selectivity-config-name CA3c \
    --arena-id A --trajectory-id Diag --use-coreneuron \
    --target-features-path $SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_features_20220131_compressed.h5 \
    --target-features-namespace 'Place Selectivity' \
    --input-features-path $SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_features_20220131_compressed.h5 


fi
