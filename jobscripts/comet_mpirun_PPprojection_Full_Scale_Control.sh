#!/bin/bash
#
#SBATCH -J PPprojection_Full_Scale_Control
#SBATCH -o ./results/PPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -p compute
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

forest=$SLURM_ARRAY_TASK_ID 
if test "$forest" = "";
then
  forest=250
fi

forest_dir=/oasis/scratch/comet/$USER/temp_project/dentate/Full_Scale_Control/GC/$forest
gridcell_dir=/oasis/scratch/comet/$USER/temp_project/GridCellModules
results_dir=/oasis/scratch/comet/$USER/temp_project/PPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

ibrun $HOME/dentate/scripts/DGnetwork/PPprojection -t $forest_dir -p $gridcell_dir -r 8.5 --grid-cells=10:3800 -o $results_dir
