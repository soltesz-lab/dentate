#!/bin/bash
#
#SBATCH -J generate_distance_normal_weights_DG_MC
#SBATCH -o ./results/generate_normal_weights_DG_MC.%j.o
#SBATCH -N 10
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

ibrun python3 ./scripts/generate_normal_weights_as_cell_attr.py \
    -d MC -s GC -s MC \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --weights-path=$DATA_PREFIX/Full_Scale_Control/DG_MC_syn_weights_LN_20221210.h5 \
    --connections-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_connections_20221210_compressed.h5 \
    --io-size=20  --value-chunk-size=10000 --chunk-size=10000 --write-size=0 -v




