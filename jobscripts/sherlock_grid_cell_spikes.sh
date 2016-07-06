#!/bin/bash
#
#SBATCH -J grid_cell_spikes
#SBATCH -o ./results/grid_cell_spikes.%j.o
#SBATCH -t 8:00:00
#SBATCH --mem 8192
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load matlab

BATCH_SIZE=190
BATCH_INDEX=$SLURM_ARRAY_TASK_ID

export BATCH_SIZE
export BATCH_INDEX

cd $HOME/model/dentate/scripts/GridCells
./run_gen_spikes.sh /share/sw/licensed/MATLAB-R2015b


