#!/bin/bash -l

#SBATCH -N 2
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -J generate_DG_PP_spiketrains_20171105
#SBATCH -e generate_DG_PP_spiketrains_20171105.err
#SBATCH -o generate_DG_PP_spiketrains_20171105.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronmil@stanford.edu

source $HOME/.bash_profile

set -x
srun -n 128 python $HOME/dentate/scripts/generate_DG_PP_spiketrains.py --selectivity-path=$SCRATCH/Full_Scale_Control/dentate_Full_Scale_Control_coords_PP_spiketrains_20171105.h5
