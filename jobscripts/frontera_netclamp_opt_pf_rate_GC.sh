#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 2             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load intel/18.0.5
module load python3
module load phdf5

set -x

export FI_MLX_ENABLE_SPAWN=yes
export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

cd $SLURM_SUBMIT_DIR

if test "$2" == ""; then
ibrun python3 network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC -g $1 -t 9500 --n-trials 10 --trial-regime best \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 2000  --opt-epsilon 1 \
    --param-config-name 'Weight all inh soma all-dend' \
    --arena-id A --trajectory-id Diag \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20200901.h5 \
    --use-coreneuron \
    selectivity_rate
else
ibrun python3 network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC -g $1 -t 9500 --n-trials 10 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --results-file "$2" \
    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 2000  --opt-epsilon 1 \
    --param-config-name 'Weight all no MC inh soma all-dend' \
    --arena-id A --trajectory-id Diag \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20200901.h5 \
    --use-coreneuron \
    selectivity_rate
fi
