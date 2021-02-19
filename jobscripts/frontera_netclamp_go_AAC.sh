#!/bin/bash

#SBATCH -J netclamp_opt_pf_AAC # Job name
#SBATCH -o ./results/netclamp_opt_pf_AAC.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp_opt_pf_AAC.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 2             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001

module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

cd $SLURM_SUBMIT_DIR

ibrun -n 1  python3 network_clamp.py go  -c 20210218_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml \
    -p AAC -g 1042800 -t 9500 --n-trials 1 \
    --template-paths $DG_HOME/templates:$HOME/model/dgc/Mateos-Aparicio2014 \
    --dataset-prefix $SCRATCH/striped/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path /scratch1/03320/iraikov/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_selection_20210218.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --config-prefix config  --arena-id A --trajectory-id Diag
 
