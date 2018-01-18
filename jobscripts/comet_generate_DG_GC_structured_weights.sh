#!/bin/bash
#
#SBATCH -J generate_DG_GC_structured_weights
#SBATCH -o ./results/generate_DG_GC_structured_weights.%j.o
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
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/model/dentate:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so
ulimit -c unlimited

set -x

ibrun -np 1680 python $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
 -d GC -s LPP -s MPP \
 --stimulus-path=$SCRATCH/dentate/Full_Scale_Control/dentate_Full_Scale_Control_coords_PP_spiketrains_20171105.h5 --stimulus-namespace='Vector Stimulus' \
 --initial-weights-namespace='Weights' --structured-weights-namespace='Structured Weights' \
 --weights-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_weights_20171121_compressed.h5 \
 --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20171105.h5 \
 --trajectory-id=0 \
 --io-size=256


