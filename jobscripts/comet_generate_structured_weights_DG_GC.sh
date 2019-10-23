#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_GC
#SBATCH -o ./results/generate_structured_weights_DG_GC.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=20
#SBATCH -t 12:00:00
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



ibrun -np 1280 gdb -q -batch -x $HOME/gdb_script --args \
    python3.5 $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --structured-weights-namespace='Structured Weights' \
    --output-weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_syn_weights_SLN_20191007.h5 \
    --weights-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_syn_weights_LN_20190717_compressed.h5 \
    --connections-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190717_compressed.h5 \
    --input-features-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20190909_compressed.h5" \
    --arena-id=A \
    --io-size=128 --cache-size=10  --value-chunk-size=100000 --chunk-size=20000 --write-size=10 -v



