#!/bin/bash
#
#SBATCH -J distribute_clustered_synapses_proximal_pf
#SBATCH -o ./results/distribute_clustered_synapses_DG_proximal_pf.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH -t 0:30:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3/3.9.2
module load phdf5/1.10.4

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export DATA_PREFIX=$SCRATCH/striped2/dentate

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6
export I_MPI_ADJUST_REDUCE=6

ibrun python3  ./scripts/distribute_clustered_synapse_locs.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --template-path=./templates \
    --arena-id A \
    -i GC \
    --forest-path=$DATA_PREFIX/Slice/dentatenet_Slice_SLN_proximal_pf_20221216.h5 \
    --output-path=$DATA_PREFIX/Slice/dentatenet_Slice_SLN_proximal_pf_20221216.h5 \
    --dry-run \
    --io-size=16 -v
