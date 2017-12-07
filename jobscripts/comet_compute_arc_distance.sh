#!/bin/bash
#
#SBATCH -J compute_arc_distance
#SBATCH -o ./results/compute_arc_distance.%j.o
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=24
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/model/dentate:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so

set -x

ibrun -np 384  python ./scripts/compute_arc_distance.py \
       --config=./config/Full_Scale_Control.yaml \
       --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_distances_20171109.h5 \
       --coords-namespace=Coordinates \
       -l OML \
       --io-size=32

