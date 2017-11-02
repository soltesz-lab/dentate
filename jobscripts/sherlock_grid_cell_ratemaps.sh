#!/bin/bash
#
#SBATCH -J grid_cell_ratemap
#SBATCH -o ./results/grid_cell_ratemap.%j.o
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --mem=63240
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load matlab

DATA_PATH=$SCRATCH/gridcells
export DATA_PATH

cd $HOME/model/dentate/scripts/SpatialCells
#./run_gen_linear_ratemap.sh /share/sw/licensed/MATLAB-R2016b

matlab -nojvm -r 'run ./gen_ec2_gridcell_ratemap.m'


