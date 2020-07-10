#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_MC
#SBATCH -o ./results/generate_structured_weights_DG_MC.%j.o
#SBATCH -N 40
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load python3
module load phdf5

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export DATA_PREFIX=$SCRATCH/striped/dentate/Full_Scale_Control

set -x

cd $SLURM_SUBMIT_DIR


ibrun python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d MC -s CA3c -n GC -n MC \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --initial-weights-path=$DATA_PREFIX/DG_MC_syn_weights_20200708.h5 \
    --non-structured-weights-namespace='Normal Weights' \
    --non-structured-weights-path=$DATA_PREFIX/DG_MC_syn_weights_20200708.h5 \
    --output-weights-namespace='Structured Weights' \
    --output-weights-path=$DATA_PREFIX/DG_MC_syn_weights_S_20200708.h5 \
    --connections-path=$DATA_PREFIX/DG_MC_connections_20200708_compressed.h5 \
    --input-features-path=$DATA_PREFIX/DG_input_features_20200611_compressed.h5 \
    --arena-id=A --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --io-size=256 --cache-size=10  --value-chunk-size=100000 --chunk-size=20000 --write-size=4 --dry-run -v

