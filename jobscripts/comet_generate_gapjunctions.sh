#!/bin/bash
#
#SBATCH -J generate_gap_junctions
#SBATCH -o ./results/generate_gap_junctions.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
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

ibrun -np 24 python ./scripts/generate_gapjunctions.py \
    --config=./config/Full_Scale_Pas_GJ.yaml \
    --types-path=./datasets/dentate_h5types_gj.h5 \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DG_IN_forest_20190325.h5 \
    --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_gapjunctions_20190424.h5 \
    --connectivity-namespace="Gap Junctions" \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20190122.h5 \
    --coords-namespace="Coordinates" \
    --io-size=4 -v
