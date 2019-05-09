#!/bin/bash
#
#SBATCH -J generate_MC_normal_weights
#SBATCH -o ./results/generate_MC_normal_weights.%j.o
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
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

ibrun -np 720  \
 python2.7 $HOME/model/dentate/scripts/generate_normal_weights_as_cell_attr.py \
    -d MC -s MC \
    --config-prefix=./config \
    --config=Full_Scale_GC_Exc_Sat_LNN.yaml \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_syn_weights_N_20190503.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20190430_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=25 -v 


