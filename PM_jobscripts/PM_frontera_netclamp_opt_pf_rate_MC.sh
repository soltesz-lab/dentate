#!/bin/bash

#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

#module load intel/18.0.5
module load python3
module load phdf5
ml load intel19
#export PYTHONPATH=/home1/04119/pmoolcha/intel18:$PYTHONPATH

set -x

export FI_MLX_ENABLE_SPAWN=yes
#export NEURONROOT=$HOME/bin/nrnpython3
#export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
#export PATH=$NEURONROOT/x86_64/bin:$PAT
export MODEL_HOME=/scratch1/04119/pmoolcha/HDM
export DG_HOME=$MODEL_HOME/dentate
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so
export RAIKOVSCRATCH=/scratch1/03320/iraikov

#cd $SLURM_SUBMIT_DIR



if test "$2" == ""; then
ibrun -n 301 -rr python3 network_clamp.py optimize  -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p MC -g 1000018 -g 1007283 -g 1016576 -g 1024018 -g 1029806 \
    -t 9500 --dt 0.001 \
    --n-trials 3 --trial-regime best \
    --template-paths $DG_HOME/templates:$MODEL_HOME/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $RAIKOVSCRATCH/striped/dentate \
    --results-path results/netclamp \
    --input-features-path $RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 400  --opt-epsilon 1 \
    --param-config-name 'Weight exc inh netclamp' \
    --arena-id A --trajectory-id Diag \
    --target-features-path $RAIKOVSCRATCH/dentate/Slice/MC_extent_input_spike_trains_20201001.h5 \
    --opt-seed 78236474 \
    rate
else
ibrun -n 301 -rr python3 network_clamp.py optimize  -c 20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    -p MC -g 1000018 -g 1007283 -g 1016576 -g 1024018 -g 1029806 \
    -t 9500 --dt 0.001 \
    --n-trials 3 --trial-regime best \
    --template-paths $DG_HOME/templates:$MODEL_HOME/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $RAIKOVSCRATCH/striped/dentate \
    --results-path results/netclamp \
    --results-file "$2" \
    --input-features-path $RAIKOVSCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 400  --opt-epsilon 1 \
    --param-config-name 'Weight exc inh netclamp' \
    --arena-id A --trajectory-id Diag \
    --target-features-path $RAIKOVSCRATCH/dentate/Slice/MC_extent_input_spike_trains_20201001.h5 \
    --opt-seed 78236474 \
    rate
fi

#    --target-rate-map-path $RAIKOVSCRATCH/dentate/Slice/MC_extent_input_spike_trains_20201001.h5 \
