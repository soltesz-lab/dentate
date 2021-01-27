#!/bin/bash
#
#SBATCH -J distribute_IN_synapses
#SBATCH -o ./results/distribute_IN_synapses.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x

ibrun -np 24 python3.5 ./scripts/distribute_synapse_locs.py \
              --distribution=poisson \
              --config=Full_Scale_Basis.yaml \
              --template-path=$PWD/templates \
              -i AAC -i BC -i MC -i HC -i HCC -i MOPP -i NGFC -i IS \
              --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_20191130_compressed.h5 \
              --output-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20191130.h5 \
              --io-size=2 -v
