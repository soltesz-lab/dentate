#!/bin/bash
#
#SBATCH -J generate_IN_coordinates
#SBATCH -o ./results/generate_IN_coordinates.%j.o
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#



module load python
module unload intel
module load gnu
module load mvapich2_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x


ibrun -np 12 python3 ./scripts/generate_soma_coordinates.py -v \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --types-path=./datasets/dentate_h5types.h5 \
    --geometry-path=./datasets/dentate_geometry.h5 \
    --template-path=./templates \
    --resolution 40 40 10 \
    -i BC \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/dentate_IN_coords_20190621.h5 \
    --output-namespace='Generated Coordinates' 

#       -i AAC -i BC -i MC -i HC -i HCC -i IS -i MOPP -i NGFC -i MPP -i LPP -i ConMC -i CA3c \
