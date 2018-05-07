#!/bin/bash
#
#SBATCH -J generate_soma_coordinates
#SBATCH -o ./results/generate_soma_coordinates.%j.o
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load mpich/3.1.4/gcc
module load gcc/4.9.1
module load python/2.7.5

export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export PYTHONPATH=$HOME/model/dentate:$HOME/bin/nrn/lib64/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x


mpirun -np 1 python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Control.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       --output-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_20180214.h5 \
       -i AAC -i BC -i MC  -i HC -i HCC -i MOPP -i NGFC -i IS -i MPP -i LPP \
       --output-namespace='Generated Coordinates' 

