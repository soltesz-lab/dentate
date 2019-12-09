#!/bin/bash
#
#SBATCH -J generate_log_normal_weights_MC
#SBATCH -o ./results/generate_log_normal_weights_MC.%j.o
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x

ibrun -np 720  \
 python3.5 $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d MC -s GC -s CA3c \
    --config-prefix=./config \
    --config=Full_Scale_GC_Exc_Sat_LNN.yaml \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_syn_weights_LN_20191130.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_connections_20191130_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=25 -v 


