#!/bin/bash

#SBATCH -J optimize_DG_network_clamp_pf # Job name
#SBATCH -o ./results/optimize_DG_network_clamp_pf.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_DG_network_clamp_pf.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 2             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load intel/18.0.5
module load python3
module load phdf5

set -x


export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

export MPLBACKEND=SVG
cd $SLURM_SUBMIT_DIR

ibrun python3 -m nested.optimize  \
    --config-file-path=$DG_HOME/config/DG_optimize_netclamp_pf.yaml \
    --output-dir=$SCRATCH/dentate/results/netclamp \
    --pop_size=112 \
    --max_iter=2000 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=1 \
    --no_cleanup \
    --gid=126002 \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH/dentate" \
    --config_prefix=$DG_HOME/config \
    --results_path=$SCRATCH/dentate/results/netclamp \
    --input_features_path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200321_compressed.h5" \
    --input_features_namespaces='Place Selectivity' \
    --input_features_namespaces='Grid Selectivity' \
    --input_features_namespaces='Constant Selectivity' \
    --param_config_name='Weight all inh soma dend' \
    --arena_id=A --trajectory_id=Diag \
    --n_trials=1 \
    --target_rate_map_path=$SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20200327.h5



