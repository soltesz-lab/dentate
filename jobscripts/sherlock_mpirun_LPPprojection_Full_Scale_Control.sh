#!/bin/bash
#
#SBATCH -J LPPprojection_Full_Scale_Control
#SBATCH -o ./results/LPPprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16384
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

echo forest = $forest

forest_prefix=/scratch/users/$USER/dentate/Full_Scale_Control/GC
lppcell_dir=/scratch/users/$USER/lppcells
results_dir=/scratch/users/$USER/LPPprojection_Full_Scale_Control_forest_${forest}_$SLURM_JOB_ID
weights_path=/scratch/users/$USER/LPPsynweights.dat

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection-forest --label="LPPtoDGC" \
       -t $forest_prefix -f $forest  -p $lppcell_dir -o $results_dir -w $weights_path \
  -r 3.0 -l 3 --pp-cells=10:3400 --pp-cell-prefix=LPPCell -:hm8192M

cd $forest
cat LPPtoDGC.*.dat > LPPtoDGC.dat
rm LPPtoDGC.*.dat
