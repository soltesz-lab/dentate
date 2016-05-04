#!/bin/bash
#
#SBATCH -J LPPBCprojection_Full_Scale_Control
#SBATCH -o ./results/LPPBCprojection_Full_Scale_Control.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH -p compute
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

workdir=/oasis/scratch/comet/$USER/temp_project
coords=$workdir/dentate/B512/Full_Scale_Control/BCcoordinates.dat
lppcell_dir=$workdir/lppcells
results_dir=$workdir/LPPBCprojection_Full_Scale_Control_$SLURM_JOB_ID

mkdir -p $results_dir
cd $results_dir

mpirun $HOME/model/dentate/scripts/DGnetwork/PPprojection --label="LPPtoBC" -f $coords  -p $lppcell_dir -o $results_dir \
 -r 400.0 --maxn=500 --pp-cells=10:3400 --pp-cell-prefix=LPPCell -:hm16384M


