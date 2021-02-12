#!/bin/bash
#
#SBATCH -J generate_normal_weights_DG_GC
#SBATCH -o ./results/generate_normal_weights_DG_GC.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=24
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

#Run the job using mpirun_rsh 
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 768 \
    `which python3` $HOME/model/dentate/scripts/generate_normal_weights_as_cell_attr.py \
    -d GC -s MC \
    --config=Full_Scale_GC_Exc_Sat.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_syn_weights_N_20200708.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20200703_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=20 -v 


