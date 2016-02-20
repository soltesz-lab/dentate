#!/bin/bash
#
#SBATCH -J PPprojection_Full_Scale_Control
#SBATCH -o ./results/PPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8192
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load openmpi/1.10.2/gcc

forest=$SLURM_ARRAY_TASK_ID
if test "$forest" = "";
then
  forest=1
fi
forest_dir=/scratch/users/$USER/dentate/Full_Scale_Control/GC/$forest
gridcell_dir=/scratch/users/$USER/gridcells/GridCellModules_1000
results_dir=/scratch/users/$USER/PPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection -t $forest_dir -p $gridcell_dir -r 6.0 -o $results_dir \
 --grid-cells=10:3800
