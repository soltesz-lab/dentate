#!/bin/bash
#
#SBATCH -J generate_distance_log_normal_weights_DG_extent
#SBATCH -o ./results/generate_log_normal_weights_DG_extent.%j.o
#SBATCH -N 1
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH


set -x

ibrun -n 16 python3 ./scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --connections-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --io-size=4  --value-chunk-size=100000 --chunk-size=20000 --write-size=0 -v

ibrun -n 16 python3 ./scripts/generate_log_normal_weights_as_cell_attr.py \
    -d MC -s CA3c \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --connections-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --io-size=4  --value-chunk-size=100000 --chunk-size=20000 --write-size=0 -v




