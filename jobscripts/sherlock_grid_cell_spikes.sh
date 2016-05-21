#!/bin/bash
#
#SBATCH -J grid_cell_spikes
#SBATCH -o ./results/grid_cell_spikes.%j.o
#SBATCH -t 6:00:00
#SBATCH --mem 8192
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load octave

BATCH_SIZE=50
BATCH_INDEX=$SLURM_ARRAY_TASK_ID

export BATCH_SIZE
export BATCH_INDEX

cd $HOME/model/dentate/scripts/GridCells
octave gen_spikes.m


