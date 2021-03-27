#!/bin/bash

#SBATCH -p development      # Queue (partition) name
#SBATCH -N 12             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5
#export PYTHONPATH=/home1/04119/pmoolcha/intel18:$PYTHONPATH
ml load intel19

set -x

export FI_MLX_ENABLE_SPAWN=yes
#export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
#export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
#export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=/scratch1/04119/pmoolcha/HDM
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export RAIKOVSCRATCH=/scratch1/03320/iraikov

#cd $SLURM_SUBMIT_DIR

mkdir -p results/netclamp/MC_20210227

export nworkers=$((12 * 24))

if test "$3" == ""; then
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p MC -t 950 -g 1000000  --n-trials 1 --trial-regime mean --problem-regime every \
    --nprocs-per-worker=1 --n-iter=3 --n-initial=400 --num-generations=200 --population-size=300 --resample-fraction 0.5 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $RAIKOVSCRATCH/striped/dentate \
    --results-path $DG_HOME/results/netclamp/MC_20210227 \
    --config-prefix config  --param-config-name "$2" --selectivity-config-name CA3c \
    --arena-id A --trajectory-id Diag --use-coreneuron \
    --target-features-path $RAIKOVSCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a_compressed.h5 \
    --target-features-namespace 'Selectivity Features' \
    --spike-events-path "$RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' 
else
mpirun -rr -n $nworkers python3 optimize_selectivity.py  -c Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p MC -t 950 -g 1000000 --n-trials 1 --trial-regime mean --problem-regime every \
    --nprocs-per-worker=1 --n-iter=1 --n-initial=400 --num-generations=200 --population-size=300 --resample-fraction 0.5 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $RAIKOVSCRATCH/striped/dentate \
    --results-path $DG_HOME/dentate/results/netclamp/MC_20210227 \
    --results-file "$3" \
    --config-prefix config --param-config-name "$2" --selectivity-config-name CA3c \
    --arena-id A --trajectory-id Diag --use-coreneuron \
    --target-features-path $RAIKOVSCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a_compressed.h5 \
    --target-features-namespace 'Selectivity Features' \
    --spike-events-path "$RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-events-namespace 'Input Spikes' \
    --spike-events-t 'Spike Train' 


fi
