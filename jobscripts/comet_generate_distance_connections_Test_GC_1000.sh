#!/bin/bash
#
#SBATCH -J generate_distance_connections_Test_GC_1000
#SBATCH -o ./results/generate_distance_connections_Test_GC_1000.%j.o
#SBATCH --nodes=8
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

ibrun -np 96 python3 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Test_GC_1000.yaml \
       --forest-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_cells_20190625.h5 \
       --connectivity-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_connections_20190625.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Test_GC_1000/DG_coords_20190625.h5 \
       --coords-namespace=Coordinates \
       --io-size=24 --cache-size=20 --write-size=40 --value-chunk-size=200000 --chunk-size=50000 -v
