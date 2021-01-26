#!/bin/bash
#
#SBATCH -J measure_distances
#SBATCH -o ./results/measure_distances.%j.o
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 5:00:00
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

ibrun -np 12 python3 ./scripts/measure_distances.py \
    --config=./config/Full_Scale_Basis.yaml \
    --geometry-path=./datasets/dentate_geometry.h5 \
    --resolution 30 30 10 \
    -i CA3c -i ConMC -i MPP -i LPP -i MC -i BC -i HC -i HCC -i NGFC -i MOPP -i IS -i AAC  \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/dentate_IN_coords_dist_20190622.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=4 \
    -v

