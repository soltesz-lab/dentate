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

module load mpich/3.1.4/gcc
module load python/2.7.5
module load hdf5

export PYTHONPATH=$PI_HOME/anaconda2/lib/python2.7:$PYTHONPATH

mpirun -np 32 python ./scripts/interpolate_DG_soma_locations.py \
       --coords-path=/scratch/users/iraikov/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_20170614.h5
