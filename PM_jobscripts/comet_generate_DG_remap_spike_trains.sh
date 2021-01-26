#!/bin/bash
#
#SBATCH -J generate_DG_remap_spike_trains
#SBATCH -o ./results/generate_DG_remap_spike_trains.%j.o
#SBATCH --nodes=10
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


ibrun -np 240 \
    gdb -q -batch -x $HOME/gdb_script --args python3.5 ./scripts/generate_DG_input_spike_trains.py \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --selectivity-path=${SCRATCH}/dentate/Full_Scale_Control/DG_input_features_remap_20191113.h5 \
    --output-path=${SCRATCH}/dentate/Full_Scale_Control/DG_remap_spike_trains_20191113.h5 \
    --io-size 24 --write-size 10000 \
     -p MPP -p LPP -p CA3c -v


