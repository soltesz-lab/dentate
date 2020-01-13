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

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

$MPIRUN python3 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Full_Scale_GC_Exc_Sat.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20190717_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20191207.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190717.h5 \
       --coords-namespace=Coordinates \
       --io-size=160 --cache-size=10 --write-size=40 --value-chunk-size=200000 --chunk-size=50000 -v
