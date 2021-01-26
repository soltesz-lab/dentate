#!/bin/bash

#SBATCH -J optimize_DG_test_network_subworlds # Job name
#SBATCH -o ./results/optimize_DG_test_network_subworlds.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_test_network_subworlds.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 32             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

#module load intel/18.0.5
#module load phdf5
#ml load intel19

set -x

#ml load intel/18.0.5
ml load intel/18.0.5
ml load intel18
#ml load intel19

export MODEL_HOME=/scratch1/04119/pmoolcha/HDM
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so

#results_path=/scratch1/04119/pmoolcha/HDM/dentate/results/optimize_DG_test_network_subworlds_$SLURM_JOB_ID
results_path=results/optimize_DG_test_network_subworlds_$SLURM_JOB_ID
export results_path
export RAIKOVSCRATCH=/scratch1/03320/iraikov

export PYTHONPATH=/scratch1/04119/pmoolcha/HDM:$PYTHONPATH

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export MPLBACKEND=SVG
#cd $SLURM_SUBMIT_DIR

ibrun python3 -m nested.optimize  \
    --config-file-path=$DG_HOME/config/DG_optimize_network_subworlds_config_dbg_sobol.yaml \
    --output-dir=$results_path \
    --pop_size=1 \
    --max_iter=1 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=896 \
    --no_cleanup \
    --param_config_name="Weight inh microcircuit" \
    --arena_id=A --trajectory_id=Diag \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$RAIKOVSCRATCH/striped/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --spike_input_path="$RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike_input_attr='Spike Train' \
    --target_rate_map_path="$RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
    --target_rate_map_namespace="Place Selectivity" \
    --max_walltime_hours=2.0 \
    --io_size=8 \
    --microcircuit_inputs \
    --use_coreneuron \
    --num_models=10 \
    -v



