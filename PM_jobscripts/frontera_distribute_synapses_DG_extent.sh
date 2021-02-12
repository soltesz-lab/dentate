#!/bin/bash
#
#SBATCH -J distribute_synapses_DG_extent
#SBATCH -o ./results/distribute_synapses_DG_extent.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH -t 0:10:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load intel/18.0.5
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel18
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel18:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

ibrun -n 16 python3  ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates --populations=GC --populations=MC \
    --forest-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --output-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20210106a.h5 \
    --io-size=4 --write-size=0 \
    --chunk-size=20000 --value-chunk-size=100000 -v
