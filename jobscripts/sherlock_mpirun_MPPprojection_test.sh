#!/bin/bash
#
#SBATCH -J MPPprojection_test
#SBATCH -o ./results/MPPprojection_test.%j.o
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8192
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

forest_prefix=/scratch/users/$USER/DGC_forest_test
gridcell_dir=/scratch/users/$USER/gridcells/GridCellModules_1000
results_dir=/scratch/users/$USER/PPprojection_test_forest_${forest}_$SLURM_JOB_ID
weights_path=/scratch/users/$USER/MPPsynweights.dat

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/src/neuro-utils/scripts/PPprojection-forest --label="MPPtoDGC" \
       -v -t $forest_prefix -f $forest  -p $gridcell_dir -o $results_dir -w $weights_path \
       -r 6.5 -l 1,2 --pp-cells=10:3800 --pp-cell-prefix=GridCell

cd $forest
cat MPPtoDGC.*.dat > MPPtoDGC.dat
rm MPPtoDGC.*.dat
