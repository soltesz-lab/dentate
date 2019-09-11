#!/bin/bash
#
#SBATCH -J generate_DG_input_selectivity_features
#SBATCH -o ./results/generate_DG_input_selectivity_features.%j.o
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=24
#SBATCH -t 4:00:00
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


ibrun -np 288 \
    python3.5 $HOME/model/dentate/scripts/generate_DG_input_selectivity_features.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --coords-path=${SCRATCH}/dentate/Full_Scale_Control/DG_coords_20190717_compressed.h5 \
    --output-path=${SCRATCH}/dentate/Full_Scale_Control/DG_input_features_20190909.h5 \
    --io-size 24 \
    -p GC -p MPP -p LPP -p CA3c -p MC -v

