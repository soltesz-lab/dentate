#!/bin/bash
#
#SBATCH -J generate_gap_junctions
#SBATCH -o ./results/generate_gap_junctions.%j.o
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH -t 2:00:00
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

ibrun -np 24 python3.5 ./scripts/generate_gapjunctions.py \
    --config=./config/Full_Scale_Basis.yaml \
    --types-path=./datasets/dentate_h5types_gj.h5 \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_20190325.h5 \
    --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_gapjunctions_20190717.h5 \
    --connectivity-namespace="Gap Junctions" \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190717.h5 \
    --coords-namespace="Coordinates" \
    --io-size=4 -v
