#!/bin/bash

#SBATCH -J netclamp_opt_pf_MC_rate_1000000_all # Job name
#SBATCH -o ./results/netclamp_opt_pf_MC_rate_1000000_all.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_opt_pf_MC_rate_1000000_all.e%j       # Name of stderr error file
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

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_core.so:/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_sequential.so
export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

#results_path=$SCRATCH/dentate/results/netclamp_opt_pf_rate_47123_$SLURM_JOB_ID
#export results_path
#mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

cd $SLURM_SUBMIT_DIR

ibrun  python3 network_clamp.py optimize  -c Network_Clamp_MC_SLN.yaml \
    -p MC -g 1000000 -t 9500 --n-trials 1 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path $SCRATCH/striped/dentate/Full_Scale_Control/DG_input_features_20200611_compressed.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --opt-iter 100 --param-config-name 'Weight no GC MC' \
    --arena-id A --trajectory-id Diag \
    --target-features-path $SCRATCH/striped/dentate/Slice/DG_MC_input_spike_trains_20200708.h5 \
    selectivity_rate
