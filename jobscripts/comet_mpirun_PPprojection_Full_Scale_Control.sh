#!/bin/bash
#
#SBATCH -J PPprojection_Full_Scale_Control
#SBATCH -o ./results/PPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load openmpi_ib/1.8.4

forest=110
forest_dir=/oasis/scratch/comet/$USER/temp_project/dentate/Full_Scale_Control/GC/$forest
grid=1
gridcell_dir=/oasis/scratch/comet/$USER/temp_project/gridcell/GridModule`printf "%02d" $grid`coordinates.dat

mkdir -p /oasis/scratch/comet/$USER/temp_project/PPprojection_Full_Scale_Control_grid_$SLURM_JOB_ID
cd /oasis/scratch/comet/$USER/temp_project/PPprojection_Full_Scale_Control_$SLURM_JOB_ID

echo "DGC forest $forest" > info.txt
echo "Grid module $grid" >> info.txt

mpirun $HOME/dentate/scripts/DGnetwork/PPprojection -t $forest_dir -p $gridcell_dir -r 5.0 
