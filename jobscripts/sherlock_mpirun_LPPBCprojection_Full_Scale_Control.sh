#!/bin/bash
#
#SBATCH -J LPPBCprojection_Full_Scale_Control
#SBATCH -o ./results/LPPBCprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16384
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load openmpi/1.10.2/gcc

coords=/scratch/users/$USER/dentate/Full_Scale_Control/B512/BCcoordinates.dat
lppcell_dir=/scratch/users/$USER/lppcells
results_dir=/scratch/users/$USER/LPPBCprojection_Full_Scale_Control_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection --label="LPPtoBC" -f $coords  -p $lppcell_dir -o $results_dir \
 -r 30.0 --pp-cells=10:3400 --pp-cell-prefix=LPPCell -:hm8192M

cd $forest
cat LPPtoBC.*.dat > LPPtoBC.dat
rm LPPtoBC.*.dat
