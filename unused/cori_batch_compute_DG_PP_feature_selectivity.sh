#!/bin/bash -l

#SBATCH -N 2
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -J compute_DG_PP_feature_selectivity_20171105
#SBATCH -e compute_DG_PP_feature_selectivity_20171105.err
#SBATCH -o compute_DG_PP_feature_selectivity_20171105.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronmil@stanford.edu

source $HOME/.bash_profile

set -x
srun -n 128 python $HOME/dentate/scripts/compute_DG_PP_feature_selectivity.py --coords-path=$SCRATCH/Full_Scale_Control/dentate_Full_Scale_Control_coords_selectivity_20171105.h5
