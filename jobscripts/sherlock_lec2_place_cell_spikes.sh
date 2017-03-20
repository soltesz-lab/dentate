#!/bin/bash
#
#SBATCH -J lec2_place_cell_spikes
#SBATCH -o ./results/lec2_place_cell_spikes.%j.o
#SBATCH -t 12:00:00
#SBATCH --mem 8192
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load matlab

BATCH_SIZE=250
BATCH_INDEX=$SLURM_ARRAY_TASK_ID
INPUT_DATA_PATH=$SCRATCH/lec2_placecells/linear_place_data.mat

export BATCH_SIZE
export BATCH_INDEX
export INPUT_DATA_PATH

cd $HOME/model/dentate/scripts/SpatialCells
./run_gen_lec2_placecell_spikes.sh /share/sw/licensed/MATLAB-R2016b


