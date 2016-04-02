#!/bin/bash
#
#SBATCH -J MPPprojection_Full_Scale_Control
#SBATCH -o ./results/MPPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=4
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

forest_prefix=$WORK/dentate/Full_Scale_Control/GC
gridcell_dir=$WORK/gridcells/GridCellModules_1000
results_dir=$WORK/MPPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

ibrun $HOME/model/dentate/scripts/DGnetwork/PPprojection --label="MPPtoDGC" -t $forest_prefix -f $forest -p $gridcell_dir -r 6.5 -l 1,2 --grid-cells=10:3800 -o $results_dir

cat MPPtoDGC.*.dat > MPPtoDGC.dat
rm MPPtoDGC.*.dat
