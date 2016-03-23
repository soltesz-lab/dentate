#!/bin/bash
#
#SBATCH -J LPPprojection_Full_Scale_Control
#SBATCH -o ./results/LPPprojection_Full_Scale_Control.%j.o
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
lppcell_dir=/scratch/users/$USER/lppcells
results_dir=/scratch/users/$USER/LPPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/LPPprojection -t $forest_dir -p $lppcell_dir -r 3.5 -o $results_dir \
 --lpp-cells=10:3400
