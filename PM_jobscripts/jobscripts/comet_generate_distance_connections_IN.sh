#!/bin/bash
#
#SBATCH -J generate_distance_connections_IN
#SBATCH -o ./results/generate_distance_connections_IN.%j.o
#SBATCH --nodes=70
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

#Run the job using mpirun_rsh
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 1680 \
    python3 ./scripts/generate_distance_connections.py \
       --config-prefix=./config \
       --config=Full_Scale_Basis.yaml \
       --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_syns_20200112_compressed.h5 \
       --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20200410_bench70_1.h5 \
       --connectivity-namespace=Connections \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
       --coords-namespace=Coordinates \
       --io-size=48 --cache-size=20 --write-size=40 -v

