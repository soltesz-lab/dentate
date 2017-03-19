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

BATCH_SIZE=871
BATCH_INDEX=$SLURM_ARRAY_TASK_ID
OUTPUT_PATH=$SCRATCH/mec3_gridcells
INPUT_DATA_PATH=$SCRATCH/mec3_gridcells/linear_mec3_grid_data.mat

export BATCH_SIZE
export BATCH_INDEX
export OUTPUT_PATH
export INPUT_DATA_PATH

cd $HOME/model/dentate/scripts/SpatialCells
./run_gen_grid_spikes.sh /share/sw/licensed/MATLAB-R2016b


