#!/bin/bash
#
#SBATCH -J generate_distance_tuning_structured_weights_DG_GC
#SBATCH -o ./results/generate_tuning_structured_weights_DG_GC.%j.o
#SBATCH -N 512
#SBATCH -n 28672
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 6:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

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
    -d GC -s MPP -s LPP \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --structured-weights-namespace='Structured Weights' \
    --output-weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_syn_weights_S_20191218.h5 \
    --h5types-path=$SCRATCH/dentate/Full_Scale_Control/dentate_h5types.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190717_compressed.h5 \
    --input-features-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20190909_compressed.h5" \
    --arena-id=A --field-width-scale=1.33 \
    --io-size=320 --cache-size=10  --value-chunk-size=100000 --chunk-size=20000 --write-size=4 -v

