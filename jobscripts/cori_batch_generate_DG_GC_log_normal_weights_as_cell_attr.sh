#!/bin/bash -l

#SBATCH -N 32
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -J generate_DG_GC_log_normal_weights_20171106
#SBATCH -e generate_DG_GC_log_normal_weights_20171106.%j.err
#SBATCH -o generate_DG_GC_log_normal_weights_20171106.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronmil@stanford.edu

source $HOME/.bash_profile

set -x
srun -n 2048 python $HOME/dentate/scripts/generate_DG_GC_log_normal_weights_as_cell_attr.py --weights-path=$SCRATCH/Full_Scale_Control/DG_GC_forest_syns_weights_20171106.h5 --connections-path=$SCRATCH/Full_Scale_Control/DG_GC_connections_20171022_compressed.h5
