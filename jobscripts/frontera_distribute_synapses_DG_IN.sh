#!/bin/bash
#
#SBATCH -J distribute_synapses_DG_IN
#SBATCH -o ./results/distribute_synapses_DG_IN.%j.o
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=56
#SBATCH -t 0:30:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export DATA_PREFIX=$SCRATCH/striped2/dentate

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun python3  ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates \
    -i NGFC -i MOPP \
    --forest-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_forest_syns_20210107_compressed.h5 \
    --output-path=$DATA_PREFIX/Full_Scale_Control/DG_IN_forest_syns_20210920.h5 \
    --io-size=40 --write-size=0 \
    --chunk-size=10000 --value-chunk-size=10000 -v

# -i MC -i AAC -i BC -i HC  -i HCC -i IS 
