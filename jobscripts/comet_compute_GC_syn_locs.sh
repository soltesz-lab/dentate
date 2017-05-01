#!/bin/bash
#
#SBATCH -J dentate_compute_GC_syn_locs
#SBATCH -o ./results/dentate_compute_GC_syn_locs.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 6:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load python
module load hdf5
module load scipy
module load mpi4py

set -x

export PYTHONPATH=/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/src/btmorph:$PYTHONPATH

nodefile=`generate_pbs_nodefile`

echo python is `which python`

mpirun_rsh -export-all -hostfile $nodefile -np 1536  \
PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH PYTHONPATH=$PYTHONPATH \
python ./scripts/compute_GC_synapse_locs.py 
