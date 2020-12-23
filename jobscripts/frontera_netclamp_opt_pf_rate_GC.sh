#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 2             # Total # of nodes 
#SBATCH --ntasks-per-node=28          # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=1

# 567808
# 28250
# 'Weight all inh soma pd-dend'

cd $SLURM_SUBMIT_DIR
 
if test "$3" == ""; then
mpirun -n 4 python3 network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC -t 28250 -g $1  --n-trials 1 --trial-regime best --problem-regime mean --nprocs-per-worker=1 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20201221 \
    --config-prefix config  --opt-iter 600  --opt-epsilon 1 \
    --param-config-name "$2" \
    --arena-id A --trajectory-id MainDiags --use-coreneuron \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20201001.h5 \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' \
    selectivity_rate
else
mpirun python3 network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC  -t 28250 -g $1 --n-trials 1 --trial-regime best --problem-regime mean --nprocs-per-worker=1 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20201221 \
    --results-file "$3" \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' \
    --config-prefix config  --opt-iter 600  --opt-epsilon 1 \
    --param-config-name "$2" \
    --arena-id A --trajectory-id MainDiags --use-coreneuron --cooperative-init  \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20201001.h5 \
    selectivity_rate
fi

#    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \


