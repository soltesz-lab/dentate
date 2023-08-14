#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_GC
#SBATCH -o ./results/generate_structured_weights_DG_GC.%j.o
#SBATCH -N 40
#SBATCH --ntasks-per-node=56
#SBATCH -p development      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN


module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/python3.10:$PYTHONPATH
export DATA_PREFIX=$SCRATCH/striped2/dentate

set -x


cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped2/dentate/Full_Scale_Control  
 
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -n MC -n ConMC \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --initial-weights-path=$DATA_PREFIX/DG_GC_syn_weights_LN_20230628.h5 \
    --non-structured-weights-namespace='Normal Weights' \
    --non-structured-weights-path=$DATA_PREFIX/DG_GC_syn_weights_N_20230628.h5 \
    --output-features-namespace='Random Place Input Features' \
    --output-features-path=$DATA_PREFIX/DG_GC_syn_weights_S_20230628.h5 \
    --output-weights-namespace='Structured Weights' \
    --output-weights-path=$DATA_PREFIX/DG_GC_syn_weights_S_20230628.h5 \
    --connections-path=$DATA_PREFIX/DG_GC_connections_20230628.h5 \
    --input-features-path=$DATA_PREFIX/DG_input_features_20220216.h5 \
    --arena-id=A --arena-margin=0.3 \
    --target-amplitude=4 \
    --io-size=40 --value-chunk-size=320000 --chunk-size=20000 --write-size=120


