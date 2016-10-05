#!/bin/bash
#
#SBATCH -J dentate_export_connectivity_mpi
#SBATCH -o ./results/dentate_export_connectivity_mpi.%j.o
#SBATCH -t 12:00:00
#SBATCH --mem 32767
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load matlab/R2015a

LD_LIBRARY_PATH=/share/sw/licensed/MATLAB-R2015a/runtime/glnxa64:/share/sw/licensed/MATLAB-R2015a/bin/glnxa64:/share/sw/licensed/MATLAB-R2015a/sys/os/glnxa64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

BATCH_SIZE=512
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID+ARRAY_OFFSET))
WORK=/scratch/users/iraikov
LOC_INPUT_FILE=$WORK/dentate/Full_Scale_Control/Locations.mat
SYN_INPUT_FILE=$WORK/dentate/Full_Scale_Control/Syn_Connections_struct.mat
GJ_INPUT_FILE=$WORK/dentate/Full_Scale_Control/GJ_Connections.mat
OUTPUT_DIR=$WORK/dentate/Full_Scale_Control/B512

export BATCH_SIZE
export BATCH_INDEX

export LOC_INPUT_FILE
export SYN_INPUT_FILE
export GJ_INPUT_FILE
export OUTPUT_DIR

echo batch index = $BATCH_INDEX

time $HOME/model/dentate/scripts/export_connectivity_mpi

