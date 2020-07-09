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

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

#Run the job using mpirun_rsh 
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 24 \
    python3 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Test_GC_1000.yaml \
       --forest-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_cells_20190625.h5 \
       --connectivity-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_connections_20200706.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Test_GC_1000/DG_coords_20190625.h5 \
       --coords-namespace=Coordinates \
       --io-size=12 --cache-size=20 --write-size=10 --value-chunk-size=200000 --chunk-size=50000 -v
