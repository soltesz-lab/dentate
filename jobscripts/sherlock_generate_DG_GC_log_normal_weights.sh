#!/bin/bash
#
#SBATCH -J generate_DG_GC_log_normal_weights
#SBATCH -o ./results/generate_DG_GC_log_normal_weights.%j.o
#SBATCH -n 512
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python/2.7.5
module load mpich/3.1.4/gcc
module load gcc/4.9.1

export PYTHONPATH=$HOME/model/dentate:$HOME/bin/nrn/lib64/python:$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/bin/hdf5/lib:$LD_LIBRARY_PATH

set -x


mpirun -np 512 python ./scripts/generate_DG_GC_log_normal_weights_as_cell_attr.py \
 --weights-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_weights_20171107_compressed.h5 \
 --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20171105.h5 \
 --io-size=128
