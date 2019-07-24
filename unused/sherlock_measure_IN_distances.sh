#!/bin/bash
#
#SBATCH -J measure_IN_distances
#SBATCH -o ./results/measure_IN_distances.%j.o
#SBATCH -n 32
#SBATCH -t 4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load gcc python/2.7.13 eigen hdf5 impi readline

export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$PI_HOME/hdf5-1.10.2/lib:$LD_LIBRARY_PATH


set -x

##              -i AAC -i BC

mpirun -np 32 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates \
              -i MOPP -i LPP -i MPP -i HC -i HCC -i MC -i NGFC -i IS \
              --coords-path=$SCRATCH/dentate/Full_Scale_Control//DG_coords_20180521.h5 \
              --io-size=2 -v

