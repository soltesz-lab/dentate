#!/bin/bash

#SBATCH -J optimize_DG_network_neg2000_neg1500um # Job name
#SBATCH -o ./results/optimize_DG_network_neg2000_neg1500um.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_network_neg2000_neg1500um.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 40             # Total # of nodes 
#SBATCH --ntasks-per-node=56 # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export FI_MLX_ENABLE_SPAWN=1

#export I_MPI_ADJUST_BARRIER=4
export I_MPI_ADJUST_SCATTER=2
export I_MPI_ADJUST_SCATTERV=2
export I_MPI_ADJUST_ALLGATHER=2
export I_MPI_ADJUST_ALLGATHERV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=4

export I_MPI_HYDRA_TOPOLIB=ipl

results_path=$SCRATCH/dentate/results/optimize_DG_network_neg2000_neg1500um
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

cd $SLURM_SUBMIT_DIR

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off

mpirun -rr -n 3 \
    python3 optimize_network.py \
    --config-path=$DG_HOME/config/DG_optimize_network_test_neg2000_neg1500um.yaml \
    --optimize-file-dir=$results_path \
    --optimize-file-name='dmosopt.optimize_network_test_neg2000_neg1500um_20210119.h5' \
    --target-features-path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --target-features-namespace="Place Selectivity" \
    --verbose \
    --nprocs-per-worker=1092 \
    --n-iter=1 \
    --num-generations=100 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/striped/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike_input_attr='Spike Train' \
    --max_walltime_hours=24.0 \
    --io_size=8 \
    --microcircuit_inputs \
    --verbose

