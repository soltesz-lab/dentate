#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_MC
#SBATCH -o ./results/generate_structured_weights_DG_MC.%j.o
#SBATCH -N 20
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -p development      # Queue (partition) name
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3/3.9.2
module load phdf5/1.10.4

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export DATA_PREFIX=$SCRATCH/striped2/dentate

set -x

cd $SLURM_SUBMIT_DIR


ibrun python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d MC -s CA3c -n GC -n MC \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --initial-weights-path=$DATA_PREFIX/Full_Scale_Control/DG_MC_syn_weights_LN_20221210_compressed.h5 \
    --non-structured-weights-namespace='Normal Weights' \
    --non-structured-weights-path=$DATA_PREFIX/Full_Scale_Control/DG_MC_syn_weights_LN_20221210_compressed.h5 \
    --output-weights-namespace='Structured Weights' \
    --output-weights-path=$DATA_PREFIX/Full_Scale_Control/DG_MC_syn_weights_S_20221210.h5 \
    --output-features-namespace='Random Place Input Features' \
    --output-features-path=$DATA_PREFIX/Full_Scale_Control/DG_MC_syn_weights_S_20221210.h5 \
    --connections-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_connections_20221210_compressed.h5 \
    --input-features-path=$DATA_PREFIX/Full_Scale_Control/DG_input_features_20220216.h5 \
    --arena-id=A --arena-margin=0.3 \
    --target-amplitude=4 \
    --io-size=48 --value-chunk-size=10000 --chunk-size=10000 --write-size=0 -v 

