#!/bin/bash
#
#SBATCH -J generate_distance_normal_weights_DG_MC
#SBATCH -o ./results/generate_normal_weights_DG_MC.%j.o
#SBATCH -N 10
#SBATCH --ntasks-per-node=56            # # of mpi tasks per node
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load python3
module load phdf5

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH

set -x


cd $SLURM_SUBMIT_DIR

ibrun python3 ./scripts/generate_normal_weights_as_cell_attr.py \
    -d MC -s GC -s MC \
    --config=Full_Scale_GC_Exc_Sat.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_MC_syn_weights_20200708.h5 \
    --connections-path=$SCRATCH/striped/dentate/Full_Scale_Control/DG_MC_connections_20200708_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=20 -v




