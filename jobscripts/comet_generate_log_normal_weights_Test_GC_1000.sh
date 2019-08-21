#!/bin/bash
#
#SBATCH -J generate_log_normal_weights_Test_GC_1000
#SBATCH -o ./results/generate_log_normal_weights_Test_GC_1000.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


#module load python
#module unload intel
#module load gnu
#module load openmpi_ib
#module load mkl
#module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x


ibrun -np 12  \
    python3.5 $HOME/model/dentate/scripts/generate_log_normal_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -s MC \
    --config=Test_GC_1000.yaml \
    --config-prefix=./config \
    --weights-path=$SCRATCH/dentate/Test_GC_1000/DG_GC_syn_weights_LN_20190820.h5 \
    --connections-path=$SCRATCH/dentate/Test_GC_1000/DG_Test_GC_1000_connections_20190625_compressed.h5 \
    --io-size=2  --value-chunk-size=100000 --chunk-size=20000 --write-size=1 -v 


