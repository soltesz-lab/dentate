#!/bin/bash
#
#SBATCH -J MPPBCprojection_Full_Scale_Control
#SBATCH -o ./results/MPPBCprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16384
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load openmpi/1.10.2/gcc

coords=/scratch/users/$USER/dentate/Full_Scale_Control/B512/BCcoordinates.dat
gridcell_dir=/scratch/users/$USER/gridcells/GridCellModules_1000
results_dir=/scratch/users/$USER/MPPBCprojection_Full_Scale_Control_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection --label="MPPtoBC" -f $coords  -p $gridcell_dir -o $results_dir \
 -r 200.0 --maxn=500 --pp-cells=10:3400 --pp-cell-prefix=GridCell -:hm16384M

