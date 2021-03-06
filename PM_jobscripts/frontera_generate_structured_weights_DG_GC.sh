#!/bin/bash
#
#SBATCH -J generate_distance_structured_weights_DG_GC
#SBATCH -o ./results/generate_structured_weights_DG_GC.%j.o
#SBATCH -N 40
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 6:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load intel/18.0.5
module load python3
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH

set -x


cd $SLURM_SUBMIT_DIR

export DATA_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control  
 
export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -n MC -n ConMC \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --initial-weights-path=$DATA_PREFIX/DG_GC_syn_weights_LN_20210107_compressed.h5 \
    --non-structured-weights-namespace='Normal Weights' \
    --non-structured-weights-path=$DATA_PREFIX/DG_GC_syn_weights_LN_20210107_compressed.h5 \
    --output-weights-namespace='Structured Weights' \
    --output-weights-path=$DATA_PREFIX/DG_GC_syn_weights_S_20210107.h5 \
    --connections-path=$DATA_PREFIX/DG_GC_connections_20210107_compressed.h5 \
    --input-features-path=$DATA_PREFIX/DG_input_features_20200910_compressed.h5 \
    --arena-id=A --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --io-size=96 --value-chunk-size=10000 --chunk-size=10000 --write-size=0 -v


