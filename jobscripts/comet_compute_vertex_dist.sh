#!/bin/bash
#
#SBATCH -J compute_vertex_dist
#SBATCH -o ./results/compute_vertex_dist.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#


module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5

export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH

ulimit -c unlimited

set -x

connectivity_path=$SCRATCH/dentate/Full_Scale_Control/DG_Connections_Full_Scale_20191016.h5
coords_path=$SCRATCH/dentate/Full_Scale_Control/DG_Cells_Full_Scale_20191016.h5

ibrun -np 240 gdb -batch -x $HOME/gdb_script --args python3 ./scripts/compute_vertex_dist.py \
    -p $connectivity_path -c $coords_path -v \
    -d GC -s AAC -s BC -s MC -s HC -s HCC -s NGFC -s MOPP -s MPP -s LPP
