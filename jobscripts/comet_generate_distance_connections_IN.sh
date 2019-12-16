#!/bin/bash
#
#SBATCH -J generate_distance_connections_IN
#SBATCH -o ./results/generate_distance_connections_IN.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --res=iraikov_2184

. $HOME/comet_env.sh

ulimit -c unlimited

set -x


ibrun -v python3 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Full_Scale_Basis.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20191130_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_test_20191216.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
       --coords-namespace=Coordinates \
       --io-size=12 --cache-size=20 --write-size=40 -v

