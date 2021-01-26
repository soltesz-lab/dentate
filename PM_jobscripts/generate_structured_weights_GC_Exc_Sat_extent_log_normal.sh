#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_GC_extent
#SBATCH -o ./results/generate_structured_weights_DG_GC_extent.%j.o
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH

set -x

cd $SLURM_SUBMIT_DIR
export DATA_PREFIX=$SCRATCH/striped/dentate

ibrun -n 8 python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -n MC -n ConMC \
    --config=./config/Network_Clamp_GC_Exc_Sat_SLN_IN_Izh_extent.yaml \
    --output-weights-namespace='Structured Weights' \
    --h5types-path=$DATA_PREFIX/Full_Scale_Control/dentate_h5types.h5 \
    --initial-weights-namespace="Log-Normal Weights" \
    --initial-weights-path=$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201223.h5 \
    --non-structured-weights-path=$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201223.h5 \
    --non-structured-weights-namespace="Normal Weights" \
    --output-weights-path=$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201223.h5 \
    --output-features-path=$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201223.h5 \
    --connections-path=$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201223.h5 \
    --input-features-path=$DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --arena-id=A  --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --io-size=1 --value-chunk-size=100000 --chunk-size=20000 --write-size=1 \
    -v --coordinates 0.0 0.0
