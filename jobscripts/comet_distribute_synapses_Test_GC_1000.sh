#!/bin/bash
#
#SBATCH -J distribute_synapses_Test_GC_1000
#SBATCH -o ./results/distribute_synapses_Test_GC_1000.%j.o
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#



module load python
module unload intel
module load gnu
module load mvapich2_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x

ibrun -np 12 python3 ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Test_GC_1000.yaml \
    -i AAC -i BC -i MC -i HC -i HCC -i MOPP -i NGFC -i IS -i GC \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:$PWD/templates \
    --forest-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_forest_20190625.h5 \
    --output-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_cells_20190625.h5 \
    --io-size=2 -v
