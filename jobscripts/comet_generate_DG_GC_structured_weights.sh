#!/bin/bash
#
#SBATCH -J generate_DG_GC_structured_weights
#SBATCH -o ./results/generate_DG_GC_structured_weights.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=24
#SBATCH -t 6:00:00
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

nodefile=`generate_pbs_nodefile`

mpirun_rsh -export-all -hostfile $nodefile -np 768  python \
 $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
 -d GC -s LPP -s MPP \
 --config=./config/Full_Scale_Control.yaml \
 --stimulus-path=$SCRATCH/dentate/Full_Scale_Control/DG_PP_features_100_20180406.h5 \
 --stimulus-namespace='Vector Stimulus' \
 --initial-weights-namespace='Weights' --structured-weights-namespace='Structured Weights' \
 --weights-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_structured_weights_20180407.h5 \
 --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_compressed_20180319.h5 \
 --stimulus-id=100 \
 --io-size=160 --value-chunk-size=100000 --chunk-size=20000 --write-size=25 -v --dry-run


