#!/bin/bash
#
#SBATCH -J ca3_placecell_linear_ratemap
#SBATCH -o ./results/ca3_placecell_linear_ratemap.%j.o
#SBATCH -t 1:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load octave

DATA_PATH=/oasis/scratch/comet/iraikov/temp_project/ca3_placecells
export DATA_PATH

cd $HOME/model/dentate/scripts/SpatialCells
octave-cli gen_linear_ca3_placecell_ratemap.m



