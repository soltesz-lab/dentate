#!/bin/bash
#
#SBATCH -J generate_log_normal_weights_MC
#SBATCH -o ./results/generate_log_normal_weights_MC.%j.o
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=16
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


. $HOME/comet_env.sh

ulimit -c unlimited

set -x

ibrun -v python3 $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d MC -s GC -s CA3c \
    --config-prefix=./config \
    --config=Full_Scale_GC_Exc_Sat_LNN.yaml \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_syn_weights_LN_20200112.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20200112_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=25 -v 


