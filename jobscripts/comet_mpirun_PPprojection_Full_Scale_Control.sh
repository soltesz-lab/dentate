#!/bin/bash
#
#SBATCH -J PPprojection_Full_Scale_Control
#SBATCH -o ./results/PPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

forest=$SLURM_ARRAY_TASK_ID 
if test "$forest" = "";
then
  forest=1
fi

echo forest = $forest

WORK=/oasis/scratch/comet/$USER/temp_project

forest_dir=$WORK/dentate/Full_Scale_Control/GC/$forest
gridcell_dir=$WORK/GridCellModules
results_dir=$WORK/PPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

ibrun $WORK/model/dentate/scripts/DGnetwork/PPprojection -t $forest_dir -p $gridcell_dir -r 7.0 \
 --grid-cells=10:3800 -o $results_dir
