#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 12             # Total # of nodes 
#SBATCH --ntasks-per-node=56          # # of mpi tasks per node
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
export FI_MLX_ENABLE_SPAWN=1

export I_MPI_HYDRA_TOPOLIB=ipl
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off

# 567808
# 28250
# 'Weight all inh soma pd-dend'

cd $SLURM_SUBMIT_DIR

mkdir -p $SCRATCH/dentate/results/netclamp/20210210_Weight_all
export nworkers=$((12 * 24))

if test "$3" == ""; then
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p GC -t 9500 -g $1  --n-trials 1 --trial-regime mean --problem-regime every \
    --nprocs-per-worker=1 --n-iter=3 --n-initial=200 --num-generations=200 --population-size=300 --resample-fraction 0.5 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20210210_Weight_all \
    --config-prefix config  --param-config-name "$2" --selectivity-config-name PP \
    --arena-id A --trajectory-id Diag --use-coreneuron \
    --target-features-path $SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a_compressed.h5 \
    --target-features-namespace 'Selectivity Features' \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' 
else
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p GC  -t 9500 -g $1 --n-trials 1 --trial-regime mean --problem-regime every \
    --nprocs-per-worker=1 --n-iter=0 --num-generations=200 --population-size=250 --resample-fraction 0.5 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp/20210210_Weight_all \
    --results-file "$3" \
    --spike-events-path "$SCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' \
    --config-prefix config --param-config-name "$2" \
    --arena-id A --trajectory-id Diag --use-coreneuron  \
    --target-features-path $SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a_compressed.h5 \
    --target-features-namespace 'Selectivity Features' 
fi

#    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
#    --input-features-namespaces 'Place Selectivity' \
#    --input-features-namespaces 'Grid Selectivity' \
#    --input-features-namespaces 'Constant Selectivity' \


