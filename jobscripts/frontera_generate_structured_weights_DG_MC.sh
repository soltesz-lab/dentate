#!/bin/bash
#
#SBATCH -J generate_distance_structured_weights_DG_MC
#SBATCH -o ./results/generate_structured_weights_DG_MC.%j.o
#SBATCH -N 64
#SBATCH -n 3584
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5/1.8.16

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

set -x


#export I_MPI_EXTRA_FILESYSTEM=enable
#export I_MPI_EXTRA_FILESYSTEM_LIST=lustre
#export I_MPI_ADJUST_ALLGATHER=4
#export I_MPI_ADJUST_ALLGATHERV=4
#export I_MPI_ADJUST_ALLTOALL=4

cd $SLURM_SUBMIT_DIR

ibrun python3 ./scripts/generate_structured_weights_as_cell_attr.py \
    -d MC -s CA3c \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --structured-weights-namespace='Structured Weights' \
    --output-weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_MC_syn_weights_SLN_20191111.h5 \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_Cells_Full_Scale_20190917.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_Connections_Full_Scale_20190917.h5 \
    --input-features-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20190909_compressed.h5" \
    --arena-id=A \
    --io-size=256 --cache-size=10  --value-chunk-size=100000 --chunk-size=20000 --write-size=4 -v

