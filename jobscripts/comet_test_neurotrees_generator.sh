#!/bin/bash
#
#SBATCH -J test_neurotrees_generator
#SBATCH -o ./results/test_neurotrees_generator.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=12
#SBATCH -t 0:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module load hdf5
module load scipy
module load mpi4py

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so

ulimit -c unlimited

set -x

ibrun -np 768 python $HOME/src/neuroh5/tests/test_neurotrees_generator.py 
