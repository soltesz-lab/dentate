#!/bin/bash
#
#SBATCH -J interpolate_DG_soma_locations
#SBATCH -o ./results/interpolate_DG_soma_locations.%j.o
#SBATCH -n 32
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

set -x

mpirun -np 32 python ./scripts/interpolate_DG_soma_locations.py \
       --coords-path=/scratch/users/iraikov/dentate/Full_Scale_Control/dentate_Sampled_Soma_Locations.h5

