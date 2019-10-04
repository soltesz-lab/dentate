#!/bin/bash
#
#SBATCH -J generate_DG_GC_log_normal_weights
#SBATCH -o ./results/generate_DG_GC_log_normal_weights.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=24
#SBATCH -t 3:00:00
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


ibrun -np 768  \
    python3.5 $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -s MC \
    --config=Full_Scale_GC_Exc_Sat_LNN.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_syn_weights_LN_20190717.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190717_compressed.h5 \
    --io-size=160  --value-chunk-size=100000 --chunk-size=20000 --write-size=20 -v 


