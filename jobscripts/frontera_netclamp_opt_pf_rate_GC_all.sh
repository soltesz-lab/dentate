#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 24             # Total # of nodes 
#SBATCH --ntasks-per-node=56          # # of mpi tasks per node
#SBATCH -t 8:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -J netclamp_opt_pf_rate_GC_all
#SBATCH -o ./results/netclamp_opt_pf_rate_GC_all.%j.o
#SBATCH -A BIR20001

module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$SCRATCH/site-packages/intel19:$HOME/model:$NEURONROOT/lib/python:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export FI_MLX_ENABLE_SPAWN=1

#export I_MPI_EXTRA_FILESYSTEM=enable
#export I_MPI_ADJUST_ALLGATHER=4
#export I_MPI_ADJUST_ALLGATHERV=4
#export I_MPI_ADJUST_ALLTOALL=4
#export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2


# 28250
#cd $SLURM_SUBMIT_DIR
## --cooperative-init  
mkdir -p $SCRATCH/dentate/results/netclamp/20201221
export ntasks=$((24 * 3))

if test "$1" == ""; then
mpirun -n $ntasks python3 network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC -t 28250 --n-trials 1 --trial-regime best --problem-regime mean --nprocs-per-worker=16 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20201221 \
    --config-prefix config  --opt-iter 600  --opt-epsilon 1 \
    --param-config-name 'Weight all inh soma pd-dend' \
    --arena-id A --trajectory-id MainDiags --use-coreneuron \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20201001.h5 \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' \
    selectivity_rate
else
mpirun -n $ntasks network_clamp.py optimize  -c Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    -p GC  -t 28250 --n-trials 1  --trial-regime best --problem-regime mean --nprocs-per-worker=16 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20201216 \
    --results-file "$1" \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' \
    --config-prefix config  --opt-iter 600  --opt-epsilon 1 \
    --param-config-name 'Weight all inh soma pd-dend' \
    --arena-id A --trajectory-id MainDiags --use-coreneuron \
    --target-rate-map-path $SCRATCH/dentate/Slice/GC_extent_input_spike_trains_20201001.h5 \
    selectivity_rate
fi

#    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \


