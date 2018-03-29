#!/bin/bash
#
#SBATCH -J generate_DG_GC_log_normal_weights
#SBATCH -o ./results/generate_DG_GC_log_normal_weights.%j.o
#SBATCH --nodes=70
#SBATCH --ntasks-per-node=24
#SBATCH -t 5:00:00
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
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so

set -x

ibrun -np 1680 python $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
 -d GC -s MPP -s LPP -s MC \
 --weights-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_weights_20180329.h5 \
 --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_compressed_20180319.h5 \
 --io-size=256 -v 


