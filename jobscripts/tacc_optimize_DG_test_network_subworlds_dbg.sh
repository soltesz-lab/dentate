#!/bin/bash

#SBATCH -J optimize_DG_test_network_subworlds_dbg # Job name
#SBATCH -o ./results/optimize_dentate_subworlds.o%j       # Name of stdout output file
#SBATCH -e ./results/optimize_dentate_subworlds.e%j       # Name of stderr error file
#SBATCH -p skx-normal      # Queue (partition) name
#SBATCH -N 8              # Total # of nodes 
#SBATCH -n 384            # Total # of mpi tasks
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5/1.8.16

set -x

export NEURONROOT=$HOME/bin/nrnpython2
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$NEURONROOT/lib/python:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

results_path=$DG_HOME/results/optimize_DG_test_network_subworlds.$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR
mkdir -p $results_path

cd tests

ibrun python2.7 -m nested.optimize  \
     --config-file-path=$DG_HOME/config/DG_test_network_subworlds_config.yaml \
     --output-dir=$results_path \
     --pop-size=2 \
     --max-iter=5 \
     --path-length=1 \
     --disp \
     --procs-per-worker=192 \
     --no-cleanup \
     --verbose \
     --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
     --dataset_prefix="$WORK/dentate" \
     --config_prefix=$DG_HOME/config \
     --results_path=$results_path \
     --cell_selection_path=$DG_HOME/datasets/DG_slice.yaml \
     --spike_input_path=$DG_HOME/results/Full_Scale_GC_Exc_Sat_LNN_3365444/dentatenet_Full_Scale_GC_Exc_Sat_LNN_results.h5 \
     --spike_input_namespace='Spike Events' \
     --max-walltime-hours=0.5
