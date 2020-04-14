#!/bin/bash

#SBATCH -J optimize_DG_test_network_subworlds # Job name
#SBATCH -o ./results/optimize_DG_test_network_subworlds.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_test_network_subworlds.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 32             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load intel
module load phdf5

set -x

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

results_path=$SCRATCH/dentate/results/optimize_DG_test_network_subworlds_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export I_MPI_EXTRA_FILESYSTEM=enable
export I_MPI_EXTRA_FILESYSTEM_LIST=lustre
export I_MPI_ADJUST_ALLGATHER=4
export I_MPI_ADJUST_ALLGATHERV=4
export I_MPI_ADJUST_ALLTOALL=4

export MPLBACKEND=SVG
cd $SLURM_SUBMIT_DIR

ibrun python3 -m nested.optimize  \
    --config-file-path=$DG_HOME/config/DG_optimize_network_subworlds_config_SLN.yaml \
    --output-dir=$results_path \
    --pop_size=2 \
    --max_iter=1 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=896 \
    --no_cleanup \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/striped/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --cell_selection_path=$DG_HOME/datasets/DG_slice_20190917.yaml \
    --spike_input_path="$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20190912_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --target_population=GC \
    --max_walltime_hours=1.9 \
    --io_size=48 \
    -v



