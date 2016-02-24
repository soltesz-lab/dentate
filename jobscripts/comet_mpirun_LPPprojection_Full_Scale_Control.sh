#!/bin/bash
#
#SBATCH -J LPPprojection_Full_Scale_Control
#SBATCH -o ./results/LPPprojection_Full_Scale_Control.%j.o
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

forest_dir=$WORK/dentate/Full_Scale_Control/GC/$forest
lppcell_dir=$WORK/lppcells/
results_dir=$WORK/LPPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

ibrun $HOME/model/dentate/scripts/DGnetwork/LPPprojection -t $forest_dir -p $lppcell_dir -r 3.0 \
 --lpp-cells=10:3400 -o $results_dir
