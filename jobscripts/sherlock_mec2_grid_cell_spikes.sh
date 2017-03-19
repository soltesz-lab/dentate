#!/bin/bash
#
#SBATCH -J grid_cell_spikes
#SBATCH -o ./results/grid_cell_spikes.%j.o
#SBATCH -t 12:00:00
#SBATCH --mem 8192
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load matlab

BATCH_SIZE=100
BATCH_INDEX=$SLURM_ARRAY_TASK_ID
DATA_PATH=$SCRATCH/gridcells

export BATCH_SIZE
export BATCH_INDEX
export DATA_PATH

cd $HOME/model/dentate/scripts/SpatialCells
./run_gen_spikes.sh /share/sw/licensed/MATLAB-R2016b


