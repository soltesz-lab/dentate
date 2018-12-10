#!/bin/bash
#
#SBATCH -J generate_GC_log_normal_weights
#SBATCH -o ./results/generate_GC_log_normal_weights.%j.o
#SBATCH -N 256
#SBATCH -q premium
#SBATCH -t 8:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#


module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel
module load python/2.7-anaconda-4.4

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/cori/2.7-anaconda-4.4/lib/python2.7/site-packages:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.6



srun -n 8192 python $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s LPP -s MPP -s MC \
    --config=./config/Full_Scale_Control.yaml \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_forest_syns_log_normal_weights_20181201.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20181128_compressed.h5 \
    --io-size=160 --cache-size=80 --value-chunk-size=200000 --chunk-size=50000 -v


