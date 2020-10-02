#!/bin/bash
#
#SBATCH -J generate_log_normal_weights_Network_Clamp_extent
#SBATCH -o ./results/generate_log_normal_weights_Network_Clamp_extent.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

export input_path=$SCRATCH/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201001.h5

set -x

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 10 \
    `which python3` $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP  \
    --config=Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    --config-prefix=./config \
    --weights-path=${input_path} \
    --connections-path=${input_path} \
    --io-size=2  --value-chunk-size=100000 --chunk-size=20000 --write-size=1 -v

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 10 \
    `which python3` $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d MC -s CA3c  \
    --config=Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    --config-prefix=./config \
    --weights-path=${input_path} \
    --connections-path=${input_path} \
    --io-size=2  --value-chunk-size=100000 --chunk-size=20000 --write-size=1 -v


