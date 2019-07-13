#!/bin/bash
#
#SBATCH -J generate_distance_connections_GC
#SBATCH -o ./results/generate_distance_connections_GC.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=12
#SBATCH -t 5:00:00
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

ibrun -np 768 python3.5 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Full_Scale_GC_Exc_Sat.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20181222_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190712.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190712.h5 \
       --coords-namespace=Coordinates \
       --io-size=160 --cache-size=10 --write-size=40 --value-chunk-size=200000 --chunk-size=50000 -v
