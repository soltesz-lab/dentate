#!/bin/bash
#
#SBATCH -J measure_distances
#SBATCH -o ./results/measure_distances.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=20
#SBATCH -t 4:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 640 python ./scripts/measure_distances.py \
    --config=./config/Full_Scale_Control.yaml \
     -i GC -i MPP -i LPP -i MC -i BC -i HC -i HCC -i NGFC -i MOPP -i IS -i AAC \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20180516.h5 \
    --coords-namespace=Coordinates \
    --io-size=24 \
    -v

