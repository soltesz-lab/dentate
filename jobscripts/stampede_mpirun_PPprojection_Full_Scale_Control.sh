#!/bin/bash
#
#SBATCH -J PPprojection_Full_Scale_Control
#SBATCH -o ./results/PPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

forests=`seq 1 100 1000`

gridcell_dir=$WORK/GridCellModules


for forest in $forests;
do
 forest_dir=$WORK/dentate/Full_Scale_Control/GC/$forest 
 results_dir=$WORK/PPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID
 mkdir -p $results_dir
 cd $results_dir
 forest_dir=$forest_dir results_dir=$results_dir \
 ibrun tacc_affinity $WORK/dentate/scripts/DGnetwork/PPprojection \
 -t $forest_dir -p $gridcell_dir -r 9.0 \
 --grid-cells=10:3800 -o $results_dir
done

