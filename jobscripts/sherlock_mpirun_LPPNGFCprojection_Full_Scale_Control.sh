#!/bin/bash
#
#SBATCH -J LPPNGFCprojection_Full_Scale_Control
#SBATCH -o ./results/LPPNGFCprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16384
#SBATCH -t 16:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load openmpi/1.10.2/gcc

coords=/scratch/users/$USER/dentate/Full_Scale_Control/B512/NGFCcoordinates.dat
lppcell_dir=/scratch/users/$USER/lppcells
results_dir=/scratch/users/$USER/LPPNGFCprojection_Full_Scale_Control_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection --label="LPPtoNGFC" -f $coords  -p $lppcell_dir -o $results_dir \
 -r 500.0 --maxn=1000 --pp-cells=10:3400 --pp-cell-prefix=LPPCell -:hm16384M


