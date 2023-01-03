#!/bin/bash

#SBATCH -p small      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -t 3:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A BIR20001
#SBATCH -J netclamp_opt_rate_NGFC
#SBATCH -o ./results/netclamp_opt_rate_NGFC.%j.o

module load phdf5/1.10.4
module load python3/3.9.2

set -x

export FI_MLX_ENABLE_SPAWN=yes

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate
export MKLROOT=/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl
export LD_PRELOAD=$MKLROOT/lib/intel64_lin/libmkl_core.so:$MKLROOT/lib/intel64_lin/libmkl_sequential.so

export UCX_TLS="knem,dc_x"

cd $SLURM_SUBMIT_DIR

ibrun -n 24  python3 network_clamp.py optimize -c Network_Clamp_GC_Exc_Sat_SynExp3NMDA2_SLN_IN_PR.yaml \
    -p NGFC -g $1 --t-max 9500 --n-trials 1 --trial-regime mean --use-coreneuron --dt 0.0125 \
    --template-paths $DG_HOME/templates \
    --dataset-prefix $SCRATCH/striped2/dentate \
    --results-path $SCRATCH/dentate/results/netclamp \
    --input-features-path $SCRATCH/striped2/dentate/Full_Scale_Control/DG_input_features_20220216.h5 \
    --input-features-namespaces 'Place Selectivity' \
    --input-features-namespaces 'Grid Selectivity' \
    --input-features-namespaces 'Constant Selectivity' \
    --phase-mod --coords-path "$SCRATCH/striped2/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5" \
    --config-prefix config  --opt-iter 2000  --opt-epsilon 0.5 \
    --param-config-name "$2" \
    --arena-id=A --trajectory-id=Diag \
    rate
