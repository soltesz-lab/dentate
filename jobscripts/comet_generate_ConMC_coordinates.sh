#!/bin/bash
#
#SBATCH -J generate_ConMC_coordinates
#SBATCH -o ./results/generate_ConMC_coordinates.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x


ibrun -np 4 python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Ext.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       --template-path=./templates \
       -i ConMC \
       --output-path=$SCRATCH/dentate/Full_Scale_Control/dentate_ConMC_coords_20190515.h5 \
       --output-namespace='Generated Coordinates' 

