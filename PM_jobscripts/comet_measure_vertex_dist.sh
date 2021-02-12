#!/bin/bash
#
#SBATCH -J measure_vertex_dist
#SBATCH -o ./results/measure_vertex_dist.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=12
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
ulimit -c unlimited

set -x

ibrun -np 120 python ./scripts/measure_vertex_dist.py -d GC \
    --coords-path=$SCRATCH/dentate/Full_Scale_Control/DG_coords_20180717.h5 \
    --connectivity-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20180813_compressed.h5 \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/DG_GC_vertex_dist_20180813.h5 \
    --cache-size=40 \
    -v

