#!/bin/bash
#
#SBATCH -J generate_DG_input_selectivity_features
#SBATCH -o ./results/generate_DG_input_selectivity_features.%j.o
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=24
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


. $HOME/comet_env.sh


set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

#Run the job using mpirun_rsh
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 288 \
    python3 $HOME/model/dentate/scripts/generate_input_selectivity_features.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --coords-path=${SCRATCH}/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --output-path=${SCRATCH}/dentate/Full_Scale_Control/DG_input_features_20200321.h5 \
    --io-size 24 \
    -v


